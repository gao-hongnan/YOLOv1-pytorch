import ast
import json
import os
import re
from shutil import copyfile

import numpy as np
import pandas as pd


from pathlib import Path
import torch
from typing import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os
import random


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def calculate_iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = (
        bbox1.cpu().detach().numpy().tolist(),
        bbox2.cpu().detach().numpy().tolist(),
    )

    area1 = bbox1[2] * bbox1[3]  # bbox1's area
    area2 = bbox2[2] * bbox2[3]  # bbox2's area

    max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        # iou = intersect / union
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(
    bboxes: Union[torch.Tensor, list],
    iou_threshold: float,
    obj_threshold: float,
    box_format: str = "midpoint",
):
    """Does Non Max Suppression given bboxes

    Args:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        obj_threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes (yolo vs pascal voc)

    Shape:
        bboxes: [S * S, 6] -> [S * S, [class_pred, prob_score, x1, y1, x2, y2]]
                If S = 7, then [49, 6]

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # bboxes: [S * S, 6] -> [S * S, [class_pred, prob_score, x1, y1, x2, y2]]
    bboxes = [box for box in bboxes if box[1] > obj_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    if len(bboxes_after_nms) == 0:
        bboxes_after_nms = torch.tensor([])
    else:
        bboxes_after_nms = torch.stack(bboxes_after_nms, dim=0)
    return bboxes_after_nms


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def voc2coco(
    bboxes: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert pascal_voc to coco format.

    voc  => [xmin, ymin, xmax, ymax]
    coco => [xmin, ymin, w, h]

    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].

    Returns:
        coco_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].
    """

    # careful in place can cause mutation
    bboxes[..., 2:4] -= bboxes[..., 0:2]

    return bboxes


def coco2voc(
    bboxes: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert coco to pascal_voc format.

    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]


    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].

    Returns:
        voc_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].
    """

    # careful in place can cause mutation
    bboxes[..., 2:4] += bboxes[..., 0:2]

    return bboxes


def voc2yolo(
    bboxes: Union[np.ndarray, torch.Tensor],
    height: int = 720,
    width: int = 1280,
) -> Union[np.ndarray, torch.Tensor]:
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    # otherwise all value will be 0 as voc_pascal dtype is np.int
    # bboxes = bboxes.copy().astype(float)

    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height

    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]

    bboxes[..., 0] += bboxes[..., 2] / 2
    bboxes[..., 1] += bboxes[..., 3] / 2

    return bboxes


def yolo2voc(
    bboxes: Union[np.ndarray, torch.Tensor],
    height: int = 720,
    width: int = 1280,
) -> Union[np.ndarray, torch.Tensor]:
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    """

    # otherwise all value will be 0 as voc_pascal dtype is np.int
    # bboxes = bboxes.copy().astype(float)

    bboxes[..., 0] -= bboxes[..., 2] / 2
    bboxes[..., 1] -= bboxes[..., 3] / 2

    bboxes[..., 2] += bboxes[..., 0]
    bboxes[..., 3] += bboxes[..., 1]

    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height

    return bboxes


def voc2vocn(
    bboxes: Union[np.ndarray, torch.Tensor], height: float, width: float
) -> Union[np.ndarray, torch.Tensor]:
    """Converts from [x1, y1, x2, y2] to normalized [x1, y1, x2, y2].
    Normalized coordinates are w.r.t. original image size.
    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalized [x1, y1, x2, y2] is calculated as:
    Normalized x1 = x1 / width
    Normalized y1 = y1 / height
    Normalized x2 = x2 / width
    Normalized y2 = y2 / height
    Example:
        >>> a = xyxy2xyxyn(inputs=np.array([[1.0, 2.0, 30.0, 40.0]]), height=100, width=200)
        >>> a
        array([[0.005, 0.02, 0.15, 0.4]])
    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.
    Returns:
        (np.ndarray): Bounding boxes with the format `normalized (top left x,
        top left y, bottom right x, bottom right y)`.
    """

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / height

    return bboxes


def vocn2voc(bboxes: Union[np.ndarray, torch.Tensor], height: float, width: float):
    """Converts from normalized [x1, y1, x2, y2] to [x1, y1, x2, y2].
    Normalized coordinates are w.r.t. original image size.
    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalized [x1, y1, x2, y2] is calculated as:
    Normalized x1 = x1 / width
    Normalized y1 = y1 / height
    Normalized x2 = x2 / width
    Normalized y2 = y2 / height
    Example:
        >>> a = xyxyn2xyxy(inputs=np.array([[0.005, 0.02, 0.15, 0.4]]), height=100, width=200)
        >>> a
        array([[1., 2., 30., 40.]])
    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.
    Returns:
        (np.ndarray): Bounding boxes with the format `normalized (top left x,
        top left y, bottom right x, bottom right y)`.
    """

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * height

    return bboxes
