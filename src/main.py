import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1Darknet
from train import train_one_epoch, valid_one_epoch
from dataset import VOCDataset, get_transform
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    seed_all,
)
from loss import YoloLoss

DEBUG = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" # if macos m1
print(f"Using {DEVICE}")

if DEBUG:
    BATCH_SIZE = 4  # 64 in original paper but use 16
    NUM_WORKERS = 0  # 0 if debug
    EPOCHS = 10
    # Hyperparameters etc.
    LEARNING_RATE = 2e-5

    WEIGHT_DECAY = 0
    PIN_MEMORY = True
    SHUFFLE = False
    DROP_LAST = True
    seed_all(seed=1992)
else:
    # Hyperparameters etc.
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4  # 64 in original paper but use 16
    WEIGHT_DECAY = 0
    EPOCHS = 100
    NUM_WORKERS = 2  # 0 if debug
    PIN_MEMORY = True
    SHUFFLE = False
    DROP_LAST = True

    seed_all(seed=1992)


def main():
    model = Yolov1Darknet(
        in_channels=3, grid_size=7, num_bboxes_per_grid=2, num_classes=20
    ).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    csv_file = "./datasets/pascal_voc_128/pascal_voc_128.csv"
    images_dir = "./datasets/pascal_voc_128/images"
    labels_dir = "./datasets/pascal_voc_128/labels"

    S, B, C = 7, 2, 20

    train_transforms = get_transform(mode="train")

    valid_transforms = get_transform(mode="valid")

    voc_dataset_train = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        train_transforms,
        S,
        B,
        C,
        mode="train",
    )
    voc_dataset_valid = VOCDataset(
        csv_file,
        images_dir,
        labels_dir,
        valid_transforms,
        S,
        B,
        C,
        mode="valid",
    )

    voc_dataloader_train = DataLoader(
        dataset=voc_dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )
    voc_dataloader_valid = DataLoader(
        dataset=voc_dataset_valid,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=DROP_LAST,
    )

    #     # HN: check what is inside loader
    #     for batch in train_loader:
    #         image, label = batch
    #         print(image.shape)
    #         print(label.shape)

    #         break

    #     test_loader = DataLoader(
    #         dataset=test_dataset,
    #         batch_size=BATCH_SIZE,
    #         num_workers=NUM_WORKERS,
    #         pin_memory=PIN_MEMORY,
    #         shuffle=False,
    #         drop_last=True,
    #     )

    for epoch in range(EPOCHS):

        # pred_boxes, target_boxes = get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4
        # )

        # mean_avg_prec = mean_average_precision(
        #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        # )
        # print(f"epoch: {epoch} Train mAP: {mean_avg_prec}")

        # if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        if DEBUG:
            train_one_epoch(
                voc_dataloader_valid, model, optimizer, loss_fn, epoch, DEVICE
            )
            valid_one_epoch(
                voc_dataloader_valid, model, optimizer, loss_fn, epoch, DEVICE
            )
        else:
            train_one_epoch(
                voc_dataloader_train, model, optimizer, loss_fn, epoch, DEVICE
            )
            valid_one_epoch(
                voc_dataloader_train, model, optimizer, loss_fn, epoch, DEVICE
            )


if __name__ == "__main__":
    main()
