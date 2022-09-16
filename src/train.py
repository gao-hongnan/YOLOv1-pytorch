"""
Main file for training Yolo model on Pascal VOC dataset
"""

from cProfile import label
import torch
import torchvision.transforms as T
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    yolo2voc,
)
from loss import YoloLoss
from typing import *
import time
from dataset import cellboxes_to_boxes
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from PIL import Image, ImageDraw, ImageFont

plt.rcParams["savefig.bbox"] = "tight"

voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

voc_classes_map = {v: k for v, k in enumerate(voc_classes)}


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FT.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def train_one_epoch(
    train_loader, model, optimizer, criterion, epoch, device
) -> List[float]:
    """Train the model for one epoch.

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        epoch (_type_): _description_

    Shapes:
        inputs: (batch_size, 3, 448, 448)
        y_trues: (batch_size, 7, 7, 30)
        y_preds: (batch_size, 7 * 7 * 30) = (batch_size, 1470)

    Note:
        It is worth noting that y_preds is reshaped from (batch_size, 7, 7, 30) to (batch_size, 7 * 7 * 30).

    Returns:
        List[float]: _description_
    """
    # torch.Size([16, 3, 448, 448]) torch.Size([16, 7, 7, 30])
    model.train()
    start = time.time()
    train_bar = tqdm(train_loader, leave=True)
    train_loss_epoch_history = []
    train_loss = 0

    for batch_idx, (inputs, y_trues) in enumerate(train_bar):
        # inputs: (batch_size, 3, 448, 448)
        # y_trues: (batch_size, 7, 7, 30)
        inputs, y_trues = inputs.to(device), y_trues.to(device)

        # y_preds: (batch_size, 7 * 7 * 30) -> (batch_size, 1470)
        y_preds = model(inputs)

        loss = criterion(y_preds, y_trues)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # update progress bar
        train_bar.set_postfix(loss=loss.item())

    # if want add grid of images see:  https://github.com/ivanwhaf/yolov1-pytorch/blob/b7df7740bfa9326f3d84b7c10b4ec4ee03d607c0/train.py
    # TODO: add a function to plot image grids
    average_train_loss_per_epoch = train_loss / len(train_loader)
    print(f"Train Epoch: {epoch}")
    print(f"Mean loss: {average_train_loss_per_epoch}")
    print(f"Time Spent: {time.time() - start}s")
    train_loss_epoch_history.append(average_train_loss_per_epoch)

    return train_loss_epoch_history


# @torch.inference_mode()
def valid_one_epoch(
    valid_loader, model, optimizer, criterion, epoch, device
) -> List[float]:
    # same as train_one_epoch
    model.eval()
    start = time.time()

    valid_bar = tqdm(valid_loader, leave=True)
    valid_loss_epoch_history = []
    valid_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, y_trues) in enumerate(valid_bar):
            # inputs: (batch_size, 3, 448, 448)
            # y_trues: (batch_size, 7, 7, 30)
            inputs, y_trues = inputs.to(device), y_trues.to(device)

            optimizer.zero_grad()

            # y_preds: (batch_size, 7 * 7 * 30) -> (batch_size, 1470)
            y_preds = model(inputs)

            loss = criterion(y_preds, y_trues)
            valid_loss += loss.item()

            # update progress bar
            valid_bar.set_postfix(loss=loss.item())

            if batch_idx == 0:
                # DECODE

                # y_trues_decoded_yolo_format: (bs, S * S, 6) -> 6 is [class, obj_conf, x, y, w, w]
                # y_trues_decoded_voc_format: (bs, S * S, 4)
                # y_preds_decoded_yolo_format: (bs, S * S, 6)
                # y_preds_decoded_voc_format: (bs, S * S, 4)

                y_trues_decoded_yolo_format = cellboxes_to_boxes(
                    y_trues.detach().cpu()
                )
                # note yolo2voc expects [x, y, w, h] format so need slice
                y_trues_decoded_voc_format = yolo2voc(
                    y_trues_decoded_yolo_format[..., 2:], height=448, width=448
                )

                y_preds_decoded_yolo_format = cellboxes_to_boxes(
                    y_preds.detach().cpu()
                )
                y_preds_decoded_voc_format = yolo2voc(
                    y_preds_decoded_yolo_format[..., 2:], height=448, width=448
                )

                inputs = inputs.detach().cpu()
                image_grid = []
                # TODO: HONGNAN: remember find a way to turn TOTENSOR back to proper uint8 image? how?
                for (
                    input,
                    y_true_decoded_voc_format,
                    y_pred_decoded_yolo_format,
                    y_pred_decoded_voc_format,
                ) in zip(
                    inputs,
                    y_trues_decoded_voc_format,
                    y_preds_decoded_yolo_format,
                    y_preds_decoded_voc_format,
                ):
                    # FIXME: find way to turn Tensor back to uint8 image without using this.
                    # input: (3, 448, 448)
                    input = torch.from_numpy(
                        np.asarray(FT.to_pil_image(input))
                    ).permute(2, 0, 1)

                    # nms_bbox_pred: (N, 6) where N is number of bboxes after nms
                    #                       and 6 is [class, obj_conf, x, y, w, h]
                    nms_bbox_pred = non_max_suppression(
                        y_pred_decoded_yolo_format,
                        iou_threshold=0.5,
                        obj_threshold=0.4,
                        box_format="midpoint",
                    )

                    num_bboxes_after_nms = nms_bbox_pred.shape[0]

                    if num_bboxes_after_nms == 0:
                        # if no bboxes after nms, then just plot the image
                        # but in order for consistency, we just pass empty colors
                        # and class names to torchvision.utils.draw_bounding_boxes
                        # so it can just plot the original image instead of using continue.
                        colors = []
                        class_names = []
                    else:
                        class_names = [
                            voc_classes_map[int(class_idx.item())]
                            for class_idx in nms_bbox_pred[:, 0]
                        ]
                        colors = ["red"] * num_bboxes_after_nms

                        # font_path = os.path.join(
                        #     cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf"
                        # )
                    font_path = "./07558_CenturyGothic.ttf"

                    # num_colors =
                    # print(f"nms: {nms_bbox_pred[...,2:]}")
                    overlayed_image_true = torchvision.utils.draw_bounding_boxes(
                        input,
                        y_true_decoded_voc_format,
                        # colors=colors,
                        width=6,
                        # labels=class_names,
                    )
                    overlayed_image_pred = (
                        torchvision.utils.draw_bounding_boxes(
                            input,
                            nms_bbox_pred[..., 2:],
                            colors=colors,
                            width=6,
                            labels=class_names,
                            font_size=30,
                            font=font_path,
                        )
                    )
                    image_grid.append(overlayed_image_true)
                    image_grid.append(overlayed_image_pred)
                grid = torchvision.utils.make_grid(image_grid)
                # show(grid)
                # print(f"shape of overlayed_images: {overlayed_images.shape}")
                fig = plt.figure(figsize=(30, 30))
                plt.imshow(grid.numpy().transpose(1, 2, 0))

                plt.savefig(f"{epoch}_batch0.png", dpi=300)
                # plt.show()

                # END DECODE

    average_valid_loss_per_epoch = valid_loss / len(valid_loader)
    print(f"Valid Epoch: {epoch}")
    print(f"Mean loss: {average_valid_loss_per_epoch}")
    print(f"Time Spent: {time.time() - start}")

    valid_loss_epoch_history.append(average_valid_loss_per_epoch)
    return average_valid_loss_per_epoch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
