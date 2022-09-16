"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.

Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

import torch
import torch.nn as nn
import torchinfo

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # very important https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/62
        self.batchnorm = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        grid_size: int = 7,
        num_bboxes_per_grid: int = 2,
        num_classes: int = 20,
    ):
        """From Aladdin's repo:
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py

        TODO: revamp the code with better docstrings but for now it is ok.

        Args:
            in_channels (int): Incoming channels of image. Defaults to 3.
            grid_size (int): The grid size. Defaults to 7.
            num_bboxes_per_grid (int): The number of bbox per grid cell. Defaults to 2.
            num_classes (int): The number of classes. Defaults to 20.
        """
        # S, B, C
        super().__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S = grid_size
        self.B = num_bboxes_per_grid
        self.C = num_classes
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(self.S, self.B, self.C)

    def forward(self, x):
        x = self.darknet(x)

        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            # max pooling
            elif isinstance(x, str) and x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, S: int, B: int, C: int) -> torch.nn.Sequential:
        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )


if __name__ == "__main__":
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    batch_size = 16
    image_size = 448
    in_channels = 3
    S = 7
    B = 2
    C = 20

    x = torch.zeros(batch_size, in_channels, image_size, image_size)
    y_trues = torch.zeros(batch_size, S, S, B * 5 + C)
    yolov1 = Yolov1(
        in_channels=in_channels,
        grid_size=S,
        num_bboxes_per_grid=B,
        num_classes=C,
    )
    y_preds = yolov1(x)
    assert (
        y_preds.shape
        == (batch_size, 7 * 7 * (20 + 2 * 5))
        == (batch_size, S * S * (C + B * 5))
    )
    print(y_preds.shape)

    torchinfo.summary(
        yolov1, input_size=(batch_size, in_channels, image_size, image_size)
    )
