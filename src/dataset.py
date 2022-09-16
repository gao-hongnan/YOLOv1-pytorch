"""Points to note:
1. You need to use collate_fn if you are returning bboxes straight: https://discuss.pytorch.org/t/
   dataloader-collate-fn-throws-runtimeerror-stack-expects-each-tensor-to-be-equal-size-in-response-
   to-variable-number-of-bounding-boxes/117952/3

2. Train and validation column in csv is hardcoded, in actual training need to be random split.
3. Note that this implementation assumes S=7, B=2, C=20 and if you change then the code will break since
tensor slicing are performed based on this assumption.
"""
import torch
import os
import pandas as pd
from PIL import Image
from typing import Tuple
import numpy as np
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from utils import yolo2voc
import math

# TODO: see the mzbai's repo to collate fn properly so can use albu!


def get_transform(mode: str, image_size: int = 448) -> T.Compose:
    """Create a torchvision transform for the given mode.

    Note:
        You can append more transforms to the list if you want.
        For simplicity of this exercise, we will hardcode the transforms.

    Args:
        mode (str): The mode of the dataset, one of [train, valid, test].
        image_size (int, optional): The image size to resize to. Defaults to 448.

    Returns:
        T.Compose: The torchvision transform pipeline.
    """
    transforms = []
    # transforms.append(T.PILToTensor())
    # this is must need if not will have error or use TOTensor.
    # transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.Resize((image_size, image_size)))
    transforms.append(T.ToTensor())

    # transforms.append(
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # )

    if mode == "train":
        # do nothing for now as we want to ensure there is not flipping.
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass

    return T.Compose(transforms)


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        labels_dir: str,
        transforms,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        mode: str = "train",
    ) -> None:
        """Dataset class for Pascal VOC format.

        Args:
            csv_file (str): The path of the csv.
            images_dir (str): The path of the images directory.
            labels_dir (str): The path of the labels directory.
            transforms (_type_): The transform function
            S (int): Grid Size. Defaults to 7.
            B (int): Number of Bounding Boxes per Grid. Defaults to 2.
            C (int): Number of Classes. Defaults to 20.
            mode (str): The mode of the dataset. Defaults to "train". Must be one of
                        ["train", "valid", "test"]
        """

        self.csv_file = csv_file
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.S = S
        self.B = B
        self.C = C
        self.mode = mode

        # train/valid/test dataframe
        self.df = self.get_df()

    def get_df(self) -> pd.DataFrame:
        """This method returns the train/valid/test dataframe according to the mode.

        Returns:
            pd.DataFrame: The dataframe.
        """

        df = pd.read_csv(self.csv_file)

        if self.mode == "train":
            return df[df["train_flag"] == self.mode].reset_index(drop=True)

        if self.mode == "valid":
            return df[df["train_flag"] == self.mode].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"index: {index}")
        image_path = os.path.join(
            self.images_dir, self.df.loc[index, "image_id"]
        )

        image = Image.open(image_path).convert("RGB")

        label_path = os.path.join(
            self.labels_dir, self.df.loc[index, "label_id"]
        )

        bboxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                bboxes.append([class_label, x, y, width, height])

        # convert to tensor with no autograd history

        if self.transforms:
            image = self.transforms(image)
            bboxes = torch.tensor(bboxes, dtype=torch.float)

        # bboxes: [num_bbox, 5] where 5 is [class_id, x, y, width, height] yolo format
        bboxes = encode(bboxes, self.S, self.B, self.C)
        # bboxes = xywhn_to_label_matrix(bboxes, self.S, self.B, self.C)
        return image, bboxes


def xywhn_to_label_matrix(
    bboxes: torch.Tensor, S: int, B: int, C: int
) -> torch.Tensor:
    """Convert bounding boxes from xywh to label matrix for ingestion by Yolo v1.

    Following convention:
    - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/amp/ and
    - https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89

    Label matrix is 7x7x30 where the depth 30 is:
    [x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h, objectness, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]
    where p1-p20 are the 20 classes.

    But we follow aladdinpersson https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py where we follow the reverse convention:
    [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h, objectness]

    Args:
        bboxes (torch.Tensor): bboxes in YOLO format (class_label, x_center, y_center, width, height) where coordinates are normalized to [0, 1].

    Returns:
        label_matrix (torch.Tensor): label matrix in YOLO format.
    """
    # initialize label matrix
    # label_matrix has shape (S, S, 5 * B + C) = (7, 7, 30)
    label_matrix = torch.zeros((S, S, 5 * B + C))

    for each_bbox in bboxes:

        # unpack yolo bbox
        class_label, x_center, y_center, width, height = each_bbox.tolist()

        # cast class_label to int
        class_label = int(class_label)

        x_grid = math.floor(S * x_center)
        y_grid = math.floor(S * y_center)

        x_grid_offset, y_grid_offset = (
            S * x_center - x_grid,
            S * y_center - y_grid,
        )

        print(f"class_id: {class_label}")
        print(f"x_center: {x_center} y_center: {y_center}")
        print(f"width: {width} height: {height}")
        print(f"x_grid: {x_grid} y_grid: {y_grid}")
        print(f"x_grid_offset: {x_grid_offset} y_grid_offset: {y_grid_offset}")

        if (
            label_matrix[y_grid, x_grid, 20] == 0
            and label_matrix[y_grid, x_grid, 25] == 0
        ):

            label_matrix[y_grid, x_grid, 20] = 1
            label_matrix[y_grid, x_grid, 25] = 1

            new_yolo_bbox_coordinates = torch.tensor(
                [x_grid_offset, y_grid_offset, width, height]
            )

            label_matrix[y_grid, x_grid, 21:25] = new_yolo_bbox_coordinates
            label_matrix[y_grid, x_grid, 26:30] = new_yolo_bbox_coordinates

            label_matrix[y_grid, x_grid, class_label] = 1
    return label_matrix


def encode(bboxes: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:
    """Convert bounding boxes from xywh to label matrix for ingestion by Yolo v1.

    Following convention:
    - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/amp/ and
    - https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89

    Label matrix is 7x7x30 where the depth 30 is:
    [x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h, objectness, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]
    where p1-p20 are the 20 classes.

    But we follow aladdinpersson https://github.com/aladdinpersson/Machine-Learning-Collection/blob/
                                master/ML/Pytorch/object_detection/YOLO/dataset.py where we follow the reverse convention:
    [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, x_grid, y_grid, w, h, objectness, x_grid, y_grid, w, h, objectness]

    Args:
        bboxes (torch.Tensor): bboxes in YOLO format (class_label, x_center, y_center, width, height)
                               where coordinates are normalized to [0, 1].

    Returns:
        label_matrix (torch.Tensor): label matrix in YOLO format.
    """

    # initialize label_matrix
    # (S, S, 5 * B + C) -> (S, S, 30) if S=7, B=2, C=20
    label_matrix = torch.zeros((S, S, 5 * B + C))

    for bbox in bboxes:

        # unpack yolo bbox
        class_id = int(bbox[0])
        x_center, y_center, width, height = bbox[1:]

        x_grid = int(torch.floor(S * x_center))  # 当前bbox中心落在第gridx个网格,列
        y_grid = int(torch.floor(S * y_center))  # 当前bbox中心落在第gridy个网格,行

        # TODO: annotate in my documentation on exactly what this is.
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        x_grid_offset, y_grid_offset = (
            S * x_center - x_grid,
            S * y_center - y_grid,
        )

        print(f"class_id: {class_id}")
        print(f"x_center: {x_center} y_center: {y_center}")
        print(f"width: {width} height: {height}")
        print(f"x_grid: {x_grid} y_grid: {y_grid}")
        print(f"x_grid_offset: {x_grid_offset} y_grid_offset: {y_grid_offset}")

        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        # here we fill both bbox's objectness to be 1 if it is originally 0
        if (
            label_matrix[y_grid, x_grid, 20] == 0
            and label_matrix[y_grid, x_grid, 25] == 0
        ):

            label_matrix[y_grid, x_grid, 20] = 1
            label_matrix[y_grid, x_grid, 25] = 1

            encoded_bbox_coordinates = torch.tensor(
                [x_grid_offset, y_grid_offset, width, height]
            )

            label_matrix[y_grid, x_grid, 21:25] = encoded_bbox_coordinates
            label_matrix[y_grid, x_grid, 26:30] = encoded_bbox_coordinates

            # set class probability to 1
            label_matrix[y_grid, x_grid, class_id] = 1

    return label_matrix


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """
    # TODO: the first dim must be batch size so its either [bs, 1470] or [bs, 7, 7, 30]
    # in this 7x7 scenario.
    # assert (
    #     len(predictions.shape) == 4
    # ), "predictions must be a 4D tensor with batch size as first dimension"

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)),
        dim=0,
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    # why the sophistication, but either way if best_box is 0, then bboxes1 will be best_boxes, if best_box is 1, then bboxes2 will be best_boxes
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(
        predictions[..., 20], predictions[..., 25]
    ).unsqueeze(-1)

    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append(
                [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
            )
        all_bboxes.append(bboxes)
    all_bboxes = torch.tensor(all_bboxes)
    return all_bboxes


if __name__ == "__main__":
    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    S, B, C = 7, 2, 20
    mode = "train"
    train_transforms = get_transform(mode=mode)

    voc_dataset_train = VOCDataset(
        csv_file, images_dir, labels_dir, train_transforms, S, B, C, mode
    )

    print(f"Length of the dataset: {len(voc_dataset_train)}")

    for image, bboxes in voc_dataset_train:
        # print(bboxes)
        print(f"type of image: {type(image)}, type of bboxes: {type(bboxes)}")
        print(
            f"shape of image: {image.shape}, shape of bboxes: {bboxes.shape}"
        )
        print(f"bboxes: {bboxes}")

        break

    BATCH_SIZE = 4
    NUM_WORKERS = 0
    PIN_MEMORY = True
    SHUFFLE = False
    DROP_LAST = True

    train_loader = torch.utils.data.DataLoader(
        dataset=voc_dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )

    for batch_index, (images, bboxes) in enumerate(train_loader):

        images = images.detach().cpu()
        bboxes = bboxes.detach().cpu()

        print(f"images shape: {images.shape}, bboxes shape: {bboxes.shape}")

        decoded_bboxes = cellboxes_to_boxes(bboxes)
        print(f"decoded bboxes: {decoded_bboxes.shape}")
        print(f"decoded bboxes: {decoded_bboxes}")

        voc_bboxes = yolo2voc(decoded_bboxes[..., 2:], height=448, width=448)
        print(f"voc bboxes: {voc_bboxes.shape}")
        print(f"voc bboxes: {voc_bboxes}")

        image_grid = []
        for image, voc_bbox in zip(images, voc_bboxes):
            overlayed_image = torchvision.utils.draw_bounding_boxes(
                image,
                voc_bbox,
                colors=["red"] * 49,
                # labels=["dog"] * 49,
                width=6,
            )
            image_grid.append(overlayed_image)

        grid = torchvision.utils.make_grid(image_grid)
        # print(f"shape of overlayed_images: {overlayed_images.shape}")
        fig = plt.figure()
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        # plt.imshow(grid.numpy().transpose((1, 2, 0)))
        # plt.savefig(os.path.join(output_path, "batch0.png"))
        plt.show()
        # plt.close(fig)
        break
