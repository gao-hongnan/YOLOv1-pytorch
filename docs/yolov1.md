---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
mystnb:
  number_source_lines: true
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\P}{\mathbb{P}}
\newcommand{\1}{\mathbb{1}}
\newcommand{\Pobj}{\mathbb{P}(\text{obj})}
\newcommand{\yhat}{\symbf{\hat{y}}}
\newcommand{\bs}{\textbf{bs}}
\newcommand{\byolo}{\mathrm{b}_{\text{yolo}}}
\newcommand{\bgrid}{\mathrm{b}_{\text{grid}}}
\newcommand{\cc}{\mathrm{c}}
\newcommand{\iou}{\textbf{IOU}_{\symbf{\hat{b}}}^{\symbf{b}}}
\newcommand{\conf}{\textbf{conf}}
\newcommand{\confhat}{\hat{\textbf{conf}}}
\newcommand{\xx}{\mathrm{x}}
\newcommand{\yy}{\mathrm{y}}
\newcommand{\ww}{\mathrm{w}}
\newcommand{\hh}{\mathrm{h}}
\newcommand{\xxhat}{\hat{\mathrm{x}}}
\newcommand{\yyhat}{\hat{\mathrm{y}}}
\newcommand{\wwhat}{\hat{\mathrm{w}}}
\newcommand{\hhhat}{\hat{\mathrm{h}}}
\newcommand{\gx}{\mathrm{g_x}}
\newcommand{\gy}{\mathrm{g_y}}
\newcommand{\b}{\symbf{b}}
\newcommand{\bhat}{\symbf{\hat{b}}}
\newcommand{\p}{\symbf{p}}
\newcommand{\phat}{\symbf{\hat{p}}}
\newcommand{\y}{\symbf{y}}
\newcommand{\L}{\mathcal{L}}
\newcommand{\lsq}{\left[}
\newcommand{\rsq}{\right]}
\newcommand{\lpar}{\left(}
\newcommand{\rpar}{\right)}
\newcommand{\jmax}{j_{\max}}
\newcommand{\obji}{\mathbb{1}_{i}^{\text{obj}}}
\newcommand{\nobji}{\mathbb{1}_{i}^{\text{noobj}}}
\DeclareMathOperator*{\argmax}{arg\,max}
$$ 

# YOLOv1



## Notations and Definitions

### Sample Image

```{figure} https://storage.googleapis.com/reighns/images/grid_on_image.PNG
---
name: yolov1-sample-image
---
Sample Image with Grids.
```

### Bounding Box Parametrization

Given a yolo format bounding box, we will perform parametrization to transform the 
coordinates of the bounding box to a more convenient form. Before that, let us
define some notations.

````{prf:definition} YOLO Format Bounding Box
:label: yolo-bbox

The YOLO format bounding box is a 4-tuple vector consisting of the coordinates of
the bounding box in the following order:

$$
\byolo = \begin{bmatrix} \xx_c & \yy_c & \ww_n & \hh_n \end{bmatrix} \in \R^{1 \times 4}
$$ (yolo-bbox)

where 

- $\xx_c$ and $\yy_c$ are the coordinates of the center of the bounding box, 
  normalized with respect to the image width and height;
- $\ww_n$ and $\hh_n$ are the width and height of the bounding box,
  normalized with respect to the image width and height.

Consequently, all coordinates are in the range $[0, 1]$.
````

We could be done at this step and ask the model to predict the bounding box in
YOLO format. However, the author proposes a more convenient parametrization for 
the model to learn better:

1. The center of the bounding box is parametrized as the offset from the top-left
   corner of the grid cell to the center of the bounding box. We will go through an 
   an example later.

2. The width and height of the bounding box are parametrized to the square root
   of the width and height of the bounding box. 
   
````{admonition} Intuition: Parametrization of Bounding Box
The loss function of YOLOv1 is using mean squared errror.

The square root is present so that errors in small bounding boxes are more penalizing
than errors in big bounding boxes.
Recall that square root mapping expands the range of small values for values between
$0$ and $1$.

For example, if the normalized width and height of a bounding box is $[0.05, 0.8]$ respectively,
it means that the bounding box's width is 5% of the image width and height is 80% of the image height.
We can scale it back since absolute numbers are easier to visualize.

Given an image of size $100 \times 100$, the bounding box's width and height unnormalized are $5$ and $80$ respectively.
Then let's say the model predicts the bounding box's width and height to be $[0.2, 0.95]$.
The mean squared error is $(0.2 - 0.05)^2 + (0.95 - 0.8)^2 = 0.0225 + 0.0225 = 0.045$. We see that
both errors are penalized equally. But if you scale the predicted bounding box's width and height back
to the original image size, you will get $20$ and $95$ respectively, then the relative error is
much worse for the width than the height (i.e both deviates 15 pixels but the width deviates much more 
percentage wise).

Consequently, the square root mapping is used to penalize errors in small bounding boxes more than
the errors in big bounding boxes. If we use square root mapping, our original width and height
becomes $[0.22, 0.89]$ and the predicted width and height becomes $[0.45, 0.97]$. The mean squared error
is then $(0.45 - 0.22)^2 + (0.97 - 0.89)^2 = 0.0529 + 0.0064 = 0.0593$. We see that the error in the
width is penalized more than the error in the height. This helps the model to learn better by 
assigning more importance to small bounding boxes errors.
````

````{prf:definition} Parametrized Bounding Box
:label: param-bbox

The parametrized bounding box is a 4-tuple vector consisting of the coordinates of
bounding box in the following order:

$$
\b = \begin{bmatrix} f(\xx_c, \gx) & f(\yy_c, \gy) & \sqrt{\ww_n} & \sqrt{\hh_n} \end{bmatrix} \in \R^{1 \times 4}
$$ (param-bbox)

where 

- $\gx = \lfloor S \cdot \xx_c \rfloor$ is the grid cell column (row) index;
- $\gy = \lfloor S \cdot \yy_c \rfloor$ is the grid cell row (column) index;
- $f(\xx_c, \gx) = S \cdot \xx_c - \gx$ and;
- $f(\yy_c, \gy) = S \cdot \yy_c - \gy$

Take note that during construction, the square root is omitted because it is included
in the loss function later. You will see in our code later that our $\b$ is actually

$$
\begin{align}
\b &= \begin{bmatrix} f(\xx_c, \gx) & f(\yy_c, \gy) & \ww_n & \hh_n \end{bmatrix} \\
   &= \begin{bmatrix} \xx & \yy & \ww & \hh \end{bmatrix}
\end{align}
$$

We will be using the notation $[\xx, \yy, \ww, \hh]$ in the rest of the sections.

As a side note, it is often the case that a single image has multiple bounding boxes. Therefore, 
you will need to convert all of them to the parametrized form. 
````

````{prf:example} Example of Parametrization
:label: param-bbox-example

Consider the **TODO insert image** image. The bounding box is in the YOLO format at first.

$$
\byolo = \begin{bmatrix} 11 & 0.3442 & 0.611 & 0.4164 & 0.262
         \end{bmatrix}   
$$

Then since $S = 7$, we can recover $f(\xx_c, \gx)$ and $f(\yy_c, \gy)$ as follows:

$$
\begin{aligned}
\gx &= \lfloor 7 \cdot 0.3442  \rfloor &= 2 \\
\gy &= \lfloor 7 \cdot 0.611   \rfloor &= 4 \\
f(\xx_c, \gx) &= 7 \cdot 0.3442 - 2 &= 0.4093 \\
f(\yy_c, \gy) &= 7 \cdot 0.611  - 4 &= 0.2770 \\
\end{aligned}
$$

Visually, the bounding box of the dog actually lies in the 3rd column and 5th row $(3, 5)$ of the grid.
But we compute it as if it lies in the 2nd column and 4th row $(2, 4)$ of the grid because in python
the index starts from 0 and the top-left corner of the image is considered grid cell $(0, 0)$.

Then the parametrized bounding box is:

$$
\b = \begin{bmatrix} 0.4093 & 0.2770 & \sqrt{0.4164} & \sqrt{0.262} \end{bmatrix} \in \R^{1 \times 4}
$$
````

For more details, have a read at [this article](https://www.harrysprojects.com/articles/fastrcnn.html)
to understand the parametrization.

### Loss Function

See below section.

### Other Important Notations

````{prf:definition} S, B and C
:label: s-b-c


- $S$: We divide an image into an $S \times S$ grid, so $S$ is the **grid size**;
- $\gx$ denotes $x$-coordinate grid cell and $\gy$ denotes the $y$-coordinate grid cell and so the first grid cell can be denoted $(\gx, \gy) = (0, 0)$ or $(1, 1)$ if using python;
- $B$: In each grid cell $(\gx, \gy)$, we can predict $B$ number of bounding boxes;
- $C$: This is the number of classes;
- Let $\cc \in \{1, 2, \ldots, 20\}$ be a **scalar**, which is the class index (id) where
    - 20 is the number of classes;
    - in Pascal VOC: `[aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor]`
    - So if the object is class bicycle, then $\cc = 2$;
    - Note in python notation, $\cc$ starts from $0$ and ends at $19$ so need to shift accordingly.
````

````{prf:definition} Probability Object
:label: prob-object

The author defines $\Pobj$ to be the probability that an object is present in a grid cell.
This is constructed **deterministically** to be either $0$ or $1$. 

To make the notation more compact, we will add a subscript $i$ to denote the grid cell.

$$
\Pobj_i = 
\begin{cases}
    1     & \textbf{if grid cell } i \textbf{ has an object}\\
    0     & \textbf{otherwise}
\end{cases}
$$

By definition, if a ground truth bounding box's center coordinates $(\xx_c, \yy_c)$
falls in grid cell $i$, then $\Pobj_i = 1$ for that grid cell.
````


````{prf:definition} Ground Truth Confidence Score
:label: gt-confidence

The author defines the confidence score of the ground truth matrix 
to be 

$$
\conf_i = \Pobj_i \times \iou
$$

where 

$$\iou = \underset{\bhat_i \in \{\bhat_i^1, \bhat_i^2\}}{\max}\textbf{IOU}(\b_i, \bhat_i)$$

where $\bhat_i^1$ and $\bhat_i^2$ are the two bounding boxes that are predicted by the model.

It is worth noting to the readers that $\conf_i$ is also an indicator function, since
$\Pobj_i$ from {prf:ref}`prob-object` is an indicator function. 

More concretely,

$$
\conf_i =
\begin{cases}
    \textbf{IOU}(\b_i, \bhat_i)     & \textbf{if grid cell } i \textbf{ has an object}\\
    0                               & \textbf{otherwise}
\end{cases}
$$

since $\Pobj_i = 1$ if the grid cell has an object and $\Pobj_i = 0$ otherwise.

Therefore, the author is using the IOU as a proxy for the confidence score in the ground truth matrix. 
````

## Model Architecture

The model architecture from the YOLOv1 paper is presented below.

```{figure} https://storage.googleapis.com/reighns/images/yolov1_model.png
---
height: 300px
width: 600px
name: yolov1-model
---
YoloV1 Model Architecture
```

The below figure is a zoomed in version of the last layer, a cuboid of shape $7 \times 7 \times 30$.


```{figure} https://storage.googleapis.com/reighns/images/label_matrix.png
---
name: label-matrix
---
The output tensor from YOLOv1's last layer.
```

In our implementation, there are some small changes such as adding 
[**batch norm**](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) layers.
However, the overall architecture remains similar to what was proposed in the paper.

The model architecture in code is defined below:

```{code-cell} ipython3
:tags: [hide-input]

from typing import List

import torch
import torchinfo
from torch import nn

class CNNBlock(nn.Module):
    """Creates CNNBlock similar to YOLOv1 Darknet architecture

    Note:
        1. On top of `nn.Conv2d` we add `nn.BatchNorm2d` and `nn.LeakyReLU`.
        2. We set `track_running_stats=False` in `nn.BatchNorm2d` because we want
           to avoid updating running mean and variance during training.
           ref: https://tinyurl.com/ap22f8nf
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        """Initialize CNNBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            **kwargs (Dict[Any]): Keyword arguments for `nn.Conv2d` such as `kernel_size`,
                     `stride` and `padding`.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(
            num_features=out_channels, track_running_stats=False
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1Darknet(nn.Module):
    def __init__(
        self,
        architecture: List,
        in_channels: int = 3,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        init_weights: bool = False,
    ) -> None:
        """Initialize Yolov1Darknet.

        Note:
            1. `self.backbone` is the backbone of Darknet.
            2. `self.head` is the head of Darknet.
            3. Currently the head is hardcoded to have 1024 neurons and if you change
               the image size from the default 448, then you will have to change the
               neurons in the head.

        Args:
            architecture (List): The architecture of Darknet. See config.py for more details.
            in_channels (int): The in_channels. Defaults to 3 as we expect RGB images.
            S (int): Grid Size. Defaults to 7.
            B (int): Number of Bounding Boxes to predict. Defaults to 2.
            C (int): Number of Classes. Defaults to 20.
            init_weights (bool): Whether to init weights. Defaults to False.
        """
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C

        # backbone is darknet
        self.backbone = self._create_darknet_backbone()
        self.head = self._create_darknet_head()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for Conv2d, BatchNorm2d, and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        x = self.head(torch.flatten(x, start_dim=1))
        x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # if self.squash_type == "flatten":
        #     x = torch.flatten(x, start_dim=1)
        # elif self.squash_type == "3D":
        #     x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # elif self.squash_type == "2D":
        #     x = x.reshape(-1, self.S * self.S, self.C + self.B * 5)
        return x

    def _create_darknet_backbone(self) -> nn.Sequential:
        """Create Darknet backbone."""
        layers = []
        in_channels = self.in_channels

        for layer_config in self.architecture:
            # convolutional layer
            if isinstance(layer_config, tuple):
                out_channels, kernel_size, stride, padding = layer_config
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ]
                # update next layer's in_channels to be current layer's out_channels
                in_channels = layer_config[0]

            # max pooling
            elif isinstance(layer_config, str) and layer_config == "M":
                # hardcode maxpooling layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif isinstance(layer_config, list):
                conv1 = layer_config[0]
                conv2 = layer_config[1]
                num_repeats = layer_config[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels=conv1[0],
                            out_channels=conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[0]

        return nn.Sequential(*layers)

    def _create_darknet_head(self) -> nn.Sequential:
        """Create the fully connected layers of Darknet head.

        Note:
            1. In original paper this should be
                nn.Sequential(
                    nn.Linear(1024*S*S, 4096),
                    nn.LeakyReLU(0.1),
                    nn.Linear(4096, S*S*(B*5+C))
                    )
            2. You can add `nn.Sigmoid` to the last layer to stabilize training
               and avoid exploding gradients with high loss since sigmoid will
               force your values to be between 0 and 1. Remember if you do not put
               this your predictions can be unbounded and contain negative numbers even.
        """

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
            # nn.Sigmoid(),
        )
```

We then run a forward pass of the model as a sanity check.

```{code-cell} ipython3
batch_size = 4
image_size = 448
in_channels = 3
S = 7
B = 2
C = 20

DARKNET_ARCHITECTURE = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

x = torch.zeros(batch_size, in_channels, image_size, image_size)
y_trues = torch.zeros(batch_size, S, S, B * 5 + C)

yolov1 = Yolov1Darknet(
    architecture=DARKNET_ARCHITECTURE,
    in_channels=in_channels,
    S=S,
    B=B,
    C=C,
)

y_preds = yolov1(x)

print(f"x.shape: {x.shape}")
print(f"y_trues.shape: {y_trues.shape}")
print(f"y_preds.shape: {y_preds.shape}")
```

Notice how the input label `y_trues` and `y_preds` are of shape `(batch_size, S, S, B * 5 + C)`, 
in our case is `(16, 7, 7, 30)`. We will talk more in section [head](yolov1.md#head) below.

(yolov1.md#model-summary)=
### Model Summary

We use `torchinfo` package to print out the model summary.

```{code-cell} ipython3
:tags: [hide-output]

torchinfo.summary(
    yolov1, input_size=(batch_size, in_channels, image_size, image_size)
)
```

### Backbone

We use Darknet as our backbone. The backbone serves as a feature extractor.
This means that we can replace the backbone with any other feature extractor.

For example, we can replace the Darknet backbone with ResNet50, which is a 50-layer Convoluational
Neural Network. You only need to make sure that the output of the backbone can match the shape
of the input of the YOLO head. We often overcome the shape mismatch issue with
[**Global Average Pooling**](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html).

(yolov1.md#head)=
### Head

We print out the last layer of the YOLOv1 model:

```{code-cell} ipython3
:tags: [hide-output]

print(f"yolov1 last layer: {yolov1.head[-1]}")
```

Unfortunately, the info is not very helpful. We will refer back to the model summary
in section [model summary](yolov1.md#model-summary) to understand the output shape of the last layer.

Notice that the last layer is actually a linear (fc) layer with shape `[16, 1470]`. This is in
contrast of the output shape of `y_preds` which is `[16, 7, 7, 30]`. This is because in the
`forward` method, we reshaped the output of the last layer to be `[16, 7, 7, 30]`.

The reason of the reshape is because of better interpretation.
More concretely, the paper said that the core idea is to divide the input image into an $S \times S$
grid, where each grid cell has a shape of $5B + C$. See {numref}`label-matrix` for a visual example.

````{admonition} 2D matrix  vs 3D tensor
If we remove the batch size dimension, then the output tensor of `y_trues` and `y_preds` will be
a 3D tensor of shape `(S, S, B * 5 + C) = (7, 7, 30)`. 

We can also think of it as a 2D matrix, where we flatten the first and second dimension, essentially
collapsing the $7 \times 7$ grid into a single dimension of $49$. The reason why I did this is for
easier interpretation, as a 2D matrix is easier to visualize than a 3D tensor.

We will carry this idea forward in the next section.
````

## From 3D Tensor to 2D Matrix

We will now discuss how to convert the 3D tensor output of the YOLOv1 model to a 2D matrix.


```{figure} https://storage.googleapis.com/reighns/images/2d_3d.PNG
---
name: 2dto3d
---
Convert 3D tensor to 2D matrix
```

## Construction of Ground Truth Matrix

Denote the subscript $i$ in the following convention to mean the $i$-th grid cell where
$i \in \{1, 2, \ldots 49\}$ as seen in figure {numref}`2dto3d`.

We will assume $S=7$, $B=2$, and $C=20$, where

- $S$ is the grid size;
- $B$ is the number of bounding boxes to be predicted;
- $C$ is the number of classes.

Consequently, each row of the ground truth matrix will have $2B + C = 30$ elements.

````{prf:definition} YOLOv1 Ground Truth Matrix
:label: yolo_gt_matrix

Define $\y_i \in \R^{1 \times 30}$ to be the $i$-th row of the
ground truth matrix $\y \in \R^{49 \times 30}$.

$$
\y_i
= \begin{bmatrix}
\b_i & \conf_i & \b_i & \conf_i & \p_i 
\end{bmatrix} \in \R^{1 \times 30}
$$ (eq:gt_yi)

where

- $\b_i = \begin{bmatrix}\xx_i & \yy_i & \ww_i & \hh_i \end{bmatrix} \in \R^{1 \times 4}$
  as per {prf:ref}`param-bbox`;

- $\conf_i = \Pobj_i \cdot \iou \in \R$ as per {prf:ref}`gt-confidence`, note very carefully
  how $\conf_i$ is defined if the grid cell has an object, and how it is $0$ if there are no objects in
  that grid cell $i$.
  - We will keep the formal definition off the tables for now and set $\conf_i$ to be equals to $\Pobj_i$
    such that $\conf = 1$ if $\Pobj_i = 1$ and $0$ if $\Pobj_i = 0$.
  - The reason is non-trivial because we have no way of knowing the IOU of the ground truth bounding
    box with the predicted bounding box before training. You can think of it as a proxy for the
    calculation later during the loss function construction.

- $\p_i = \begin{bmatrix}0 & 0 & 1 & \cdots &0\end{bmatrix} \in \R^{1 \times 20}$ where we use the
  class id $\cc$ to construct our class probability ground truth vector such that $\p$ is everywhere 
  $0$ encoded except at the $\cc$-th index (one hot encoding). In the paper, $\p_i$ is defined as 
  $\P(\text{Class}_i \mid \text{Obj})$ which means that $\p_i$ is conditioned on the grid cell given
  there exists an object, which means for grid cells $i$ without any objects, $\p_i$ is a zero vector.

$\y_i$ will be initiated as a zero vector, and will remain a zero vector if there are no objects in
grid cell $i$. Otherwise, we will update the elements of $\y_i$ as per the above definitions.

Then the ground truth matrix $\y$ is constructed as follows:

$$
\y = \begin{bmatrix}
\y_1 \\
\y_2 \\
\vdots \\
\y_{49}
\end{bmatrix} \in \R^{49 \times 30}
$$

Note that this is often reshaped to be $\y \in \R^{7 \times 7 \times 30}$ in many implementations.
````

````{prf:remark} Remark: Ground Truth Matrix Construction
:label: yolo_gt_matrix_remark

**TODO insert encode here to show why the 1st 5 and next 5 elements are the same**

It is also worth noting to everyone that we set the first 5 elements and the next 5 elements the same,
therefore, we don't make a conscious effort to differentiate between $\b_i$,
as we will see later in the prediction matrix.
This is because we only have one set of ground truth and our choice of encoding is simply to repeat 
the ground truth coordinates twice in the first 10 elements. 

The next thing to note is that what if the same image has 2 bounding boxes having the same center coordinates?
Then **by design**, one of them will be dropped by this construction, this kind of "flawed design" 
will be fixed in future yolo iterations.

One can read how it is implemented in python under the `encode` function. The logic should follow through.

One more note is for example the dog/human image, there are two bounding boxes in that image, and one can see their center lie in different grid cells, which means the final $7 \times 7 \times 30$ matrix will have grid cell $(3, 5)$ and $(4, 4)$ filled with values of these two bounding boxes and rest are initiated with zeros since there does not exist any objects in the other grid cells. If you are using $49 \times 30$ method, then they instead like in grid cell $3 \times 7 + 5 = 26$ grid cell and $4 \times 7 + 4 = 32$ grid cell (note it is not just 3 x 5 or 4 x 4 !)

Lastly, the idea of having 2 bounding boxes in the encoding construction will be more apparent in the next section.
````

## Construction of Prediction Matrix

The construction of the prediction matrix $\hat{\y}$ follows the last layer of the neural network,
shown earlier in diagram {ref}`yolov1-model`.

To stay consistent with the shape defined in {prf:ref}`yolo_gt_matrix`, we will reshape
the last layer from $7 \times 7 \times 30$ to $49 \times 30$. As mentioned in [head section](yolov1.md#head)
the last layer is not really a 3d-tensor by design, it was in fact a linear/dense layer of shape $[-1, 1470]$.
The $1470$ neurons were reshaped to be $7 \times 7 \times 30$ so that readers like us can
interpret it better with the injection of grid cell idea.


````{prf:definition} YOLOv1 Prediction Matrix
:label: yolo_pred_matrix

Define $\hat{\y}_i \in \R^{1 \times 30}$ to be the $i$-th row of the prediction matrix
$\hat{\y} \in \R^{49 \times 30}$, output from the last layer of the neural network.

$$
\yhat_i
= \begin{bmatrix}
\bhat_i^1 & \confhat_i^1 & \bhat_i^2 & \confhat_i^2 & \phat_i 
\end{bmatrix} \in \R^{1 \times 30}
$$

where

- $\bhat_i^1 = \begin{bmatrix}\xxhat_i^1 & \yyhat_i^1 & \wwhat_i^1 & \hhhat_i^1 \end{bmatrix} 
  \in \R^{1 \times 4}$ is the predictions of the 4 coordinates made by bounding box 1;
- $\bhat_i^2$ is then the predictions of the 4 coordinates made by bounding box 2;
- $\confhat_i^1 \in \R$ is the object/bounding box confidence score (a scalar) of the first bounding
  box made by the model. As a reminder, this value will be compared during loss function with the 
  $\conf$ constructed in the ground truth;
- $\confhat_i^2 \in \R$ is the object/bounding box confidence score of the second bounding box made by the model;
- $\phat_i \in \R^{1 \times 20}$ where the model predicts a class probability vector indicating 
  which class is the most likely. By construction of loss function, this probability vector does not
  sum to 1 since the author uses MSELoss to penalize, this is slightly counter intuitive as 
  cross-entropy loss does a better job at forcing classification loss - this is remedied in later yolo versions!
    - Notice that there is no superscript for $\phat_i$, that is because the model only predicts one set of class probabilities for each grid cell $i$, even though you can predict $B$ number of bounding boxes.


Consequently, the final form of the prediction matrix $\yhat$ can be denoted as

$$
\yhat =
\begin{bmatrix}
\yhat_1 \\
\yhat_2 \\
\vdots \\
\yhat_{49}
\end{bmatrix} \in \R^{49 \times 30}
$$

and of course they must be the same shape as $\y$.
````

````{admonition} Some Remarks
:class: warning

1. Note that in our `head` layer, we did not choose to add `nn.Sigmoid()` after the last layer. This
   will cause the output of the last layer to be in the range of $[-\infty, \infty]$, which means
   it is unbounded. Therefore, non-negative values like the width and height `what_i` and `hhat_i`
   can be negative!

2. Each grid cell predicts two bound boxes, it will shift and stretch the prior box in two different ways, 
   possibly to cover two different objects (but both are constrained to have the same class). 
   You might wonder why it's trying to do two boxes. The answer is probably because 49 boxes isn't
   enough, especially when there are lots of objects close together, although what tends to happen 
   during training is that the predicted boxes become specialised. So one box might learn to find
   big things, the other might learn to find small things, this may help the network
   generalise to other domains[^1].
````

## Loss Function

Possibly the most important part of the YOLOv1 paper is the loss function, it is also
the most confusing if you are not familiar with the notation. 

````{admonition} Abuse of Notation
When I say grid cell $i$, it also means the $i$-th row of the ground truth and prediction matrix.
````

### Bipartite Matching

ref: https://www.harrysprojects.com/articles/yolov1.html

In our example image, there are 2 ground truth bounding box dog and human in that image. Let us zoom into just 1 grid cell's 30 element vector, say the dog (maybe input numbers here?) lie in the grid cell (3, 5) and therefore index is 26.

![](https://storage.googleapis.com/reighns/images/flattened_grid_cell.jpg)


The question that I had was that there is only 1 ground truth bbox coordinates for the dog $\b_{26}$, how then should we choose the which of the two predicted bounding boxes to compare to in the loss function? That is where the "matching happens", basically out of the two bounding box predicted by the model $\bhat_{26}^1$ and $\bhat_{26}^2$, only one can survive to eventually compare with the ground truth, and that is done by computing the IOU between the ground truth $\b_{26}$ with each $\bhat_{26}^1$ and $\bhat_{26}^2$ respectively and choosing the one with the highest IOU to be the survivor.

This is why in the construction of the ground truth we have this:

$$
\conf = 
\begin{cases}
    1  = 1 \times \iou       & \text{if } \textbf{grid cell i has an object}\\
    0  = 0  \times \iou      & \text{otherwise}
\end{cases}
$$

where we define the confidence score of the ground truth to be the IOU between the ground truth $\b_{26}$ and the "survivor" $\bhat_{26}$, chosen out of the two predictions, as was shown $\underset{\bhat_i \in \{\bhat_i^1, \bhat_i^2\}}{\max}\textbf{IOU}(\b_i, \bhat_i)$.

But why during our construction of the ground truth we have to put a placeholder $1$ or $0$ first? That is because before the model predictions were made, there is no way we know the IOU between the ground truth and the predicted bounding boxes.

**(Put admonition note)**

What we have described above is a form of matching algorithm. To reiterate, a model like YOLOv1 can output and predict multiple $B$ number of bounding boxes ($B=2$), but you need to choose one out of the $B$ predicted bounding boxes to compute/compare with the ground truth bounding box. In YOLOv1, they used the same matching algorithm that two-staged detectors like Faster RCNN use, which use the IOU between the ground truth and predicted bounding boxes to determine matching, (i.e ground truth in grid cell i will match to the predicted bbox in grid cell i with the highest IOU between them).

It's also worth pointing out that two-stage architectures also specify a minimum IOU for defining negative background boxes, and their loss functions explicitly ignore all predicted boxes that fall between these thresholds. YOLO doesn't do this, most likely because it's producing so few boxes anyway that it isn't a problem in practice. (https://www.harrysprojects.com/articles/yolov1.html)


### Total Loss for a Single Image

Having the construction of the ground truth and the prediction matrix, it is now time to understand
how the loss function is formulated. I took the liberty to change the notations from the original
paper for simplicity.

We define the loss function to be $\L$, a function of $\y$ and $\yhat$ respectively. 
However, owing to the fact $\y$ and $\yhat$ are both of shape $\R^{49 \times 30}$, 
it is more beneficial to take a step back and recall that we are actually computing
the loss over each grid cell $i$ and summing them (49 rows) up afterwards, which constitute
to our total loss $\L(\y, \yhat)$.

Consequently, we define $\L_i$ to be the loss of each grid cell $i$ and simply say that 

$$
\begin{align}
    \L(\y, \yhat) & \overset{(a)}{=}  \sum_{i=1}^{S=7}\sum_{j=1}^{S=7} \L_{ij}(\y_{ij}, \yhat_{ij}) \\
                  & \overset{(b)}{=} \sum_{i=1}^{S^2=49} \L_i(\y_i, \yhat_i)                        \\
\end{align}
$$ (eq:yolov1-total-loss)

but recall that the equation $(a)$ is not used by us as it is more cumbersome in notations, but just remember both are the same.

Equation {eq}`eq:yolov1-total-loss` ***merely*** sums up the loss for 1 single image, 
however, in deep learning, we also have the concept of batch size, where an additional batch 
size dimension is added. Rest assured it is as simple as summing over the batches and averaging over batch only
and will be shown in code later.

### Loss for a Single Grid Cell in a Single Image

Let's zoom in on how to calculate loss for one grid cell $i$.

$$
    \begin{align}
        \L_i(\y_i, \yhat_i) & = \color{blue}{\lambda_\textbf{coord} \sum_{j=1}^{B=2} \1_{ij}^{\text{obj}} \lsq \lpar x_i - \hat{x}_i^j \rpar^2 + \lpar y_i - \hat{y}_i^j \rpar^2 \rsq}                             \\
                            & + \color{blue}{\lambda_\textbf{coord} \sum_{j=1}^{B=2} \1_{ij}^{\text{obj}} \lsq \lpar \sqrt{w_i} - \sqrt{\hat{w}_i^j} \rpar^2 + \lpar \sqrt{h_i} - \sqrt{\hat{h}_i^j} \rpar^2 \rsq} \\
                            & + \color{green}{\sum_{j=1}^{B=2} \1_{ij}^{\text{obj}} \lpar \conf_i - \confhat_i^j \rpar^2}                                                                                          \\
                            & + \color{green}{\lambda_\textbf{noobj}\sum_{j=1}^{B=2} \1_{ij}^{\text{noobj}} \lpar \conf_i - \confhat_i^j \rpar^2}                                                                  \\
                            & + \color{red}{\obji \sum_{c \in \cc} \lpar \p_i(c) - \phat_i(c) \rpar^2}                                                                                               \\
    \end{align}
$$

- $\mathbb{1}_{i}^{obj}$ is $1$ when there is an object in cell $i$ and $0$ elsewhere
- $\mathbb{1}_{ij}^{obj}$ "denotes that the $j$th bounding box predictor in cell $i$ is responsible for that prediction". In other words, it is equal to $1$ if there is an object in cell $i$ and confidence of the $j$th predictors of this cell is the highest among all the predictors of this cell. $\mathbb{1}_{ij}^{noobj}$ is almost the same except it values 1 when there are NO objects in cell $i$

Note carefully $j$ in this context is the indices of the bounding box predictors in each grid cell i.e. in $\bhat^1$ is the 1st predicted bounding box and the 1 refers to the index $j=1$.

Do not be afraid of the $\1_{ij}^{\text{obj}}$, it simply means that in grid cell $i$, we loop over the number of bounding boxes (which is 2), and then for each predicted bounding box, we check what $\1_{ij}^{\text{obj}}$ evaluates to:

$$
\1_{ij}^{\text{obj}} = 
\begin{cases}
    1     & \textbf{if the jth box in the ith grid cell is matched with a ground truth object in grid cell i}\\
    0     & \textbf{if the jth box in the ith grid cell is not matched with a ground truth object in grid cell i}
\end{cases}
$$

and **what is matched with an object mean? -> it means EXPLAIN HERE ON BIPARTITE MATCH**

### Change of Notation

Let $\jmax$ be the index of the bounding box with the highest confidence score in cell $i$.

$\jmax = \underset{j \in \{1,2\}}{\operatorname{argmax}} \textbf{IOU}(\b_i, \bhat_i^j)$



$$
    \begin{align}
        \L_i(\y_i, \yhat_i) & \overset{(a)}{=}  \color{blue}{\lambda_\textbf{coord} \obji \lsq \lpar x_i - \hat{x}_i^{\jmax} \rpar^2 + \lpar y_i - \hat{y}_i^{\jmax}  \rpar^2 \rsq}                             \\
                            & \overset{(b)}{+}  \color{blue}{\lambda_\textbf{coord} \obji \lsq \lpar \sqrt{w_i} - \sqrt{\hat{w}_i^{\jmax} } \rpar^2 + \lpar \sqrt{h_i} - \sqrt{\hat{h}_i^{\jmax} } \rpar^2 \rsq} \\
                            & \overset{(c)}{+}  \color{green}{\obji \lpar \conf_i - \confhat_i^{\jmax} \rpar^2}                                                                                          \\
                            & \overset{(d)}{+} \color{green}{\lambda_\textbf{noobj} \nobji \lpar \conf_i - \confhat_i^{\jmax} \rpar^2}                                                                  \\
                            & \overset{(e)}{+}  \color{red}{\obji \sum_{c \in \cc} \lpar \p_i(c) - \phat_i(c) \rpar^2}                                                                                               \\
    \end{align}
$$

thereby collapsing the equation to checking only two conditions:
- $\obji$ is $1$ when there is an object in cell $i$ and $0$ elsewhere
- $\1_{i}^{\text{noobj}}$ is $1$ when there is no object in cell $i$ and $0$ elsewhere

- $\y_i$ is exactly as defined in {prf:ref}`yolo_gt_matrix`'s equation {eq}`eq:gt_yi`.


### Walkthrough

$$
\begin{bmatrix}
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0.5694051 & 0.56999993 & 0.97450423 & 0.972 & 1. & 0.5694051 & 0.56999993 & 0.97450423 & 0.972 & 1. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 1. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0.4093485 & 0.27699995 & 0.4164306 & 0.262 & 1. & 0.4093485 & 0.27699995 & 0.4164306 & 0.262 & 1. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 1. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
  0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.\\
\end{bmatrix}
$$

$$
\begin{bmatrix}
  9.25e-01 & -1.67e-01 & 1.91e-01 & 7.23e-01 & 2.14e-01 & 1.00e+00 & 1.23e-01 & 5.77e-02 & 5.62e-01 & -1.19e-01 & -5.86e-01 & -2.33e-02 & -1.02e-01 & -1.60e-01 & -4.08e-01 & 5.64e-01 & -1.63e-01 & 2.94e-01 & -2.06e-01 & -6.84e-01 & -2.15e-01 & 3.40e-01 & 4.25e-01 & 1.39e-01 & -4.16e-01 & -2.33e-01 & 2.82e-01 & 5.17e-01 & 3.94e-01 & 6.71e-02\\
  3.33e-01 & 1.90e-01 & -8.38e-02 & 4.28e-01 & -2.90e-01 & -1.90e-01 & 3.59e-01 & 3.56e-02 & -1.30e-01 & -6.84e-01 & 4.15e-01 & -7.22e-02 & -1.82e-01 & 3.25e-01 & 3.38e-02 & -9.65e-02 & -2.58e-01 & -1.80e-01 & -4.89e-01 & -1.92e-01 & 4.00e-02 & 5.50e-01 & 4.92e-01 & 2.79e-01 & -1.57e-01 & -3.29e-01 & -1.06e-01 & 2.01e-01 & -3.94e-01 & -5.36e-01\\
  -3.57e-01 & 4.45e-01 & 6.91e-01 & -3.01e-01 & -3.70e-01 & 4.30e-01 & 2.38e-01 & 4.56e-01 & -4.86e-03 & 1.13e-01 & 2.45e-01 & 2.47e-01 & 6.85e-03 & -3.48e-01 & 3.01e-01 & -7.47e-01 & -5.57e-02 & -1.94e-01 & 1.24e-01 & -4.37e-01 & 1.72e-01 & 2.39e-01 & -4.26e-01 & 2.08e-01 & 2.55e-01 & 3.35e-01 & -1.52e-01 & -5.26e-01 & -5.71e-01 & 7.32e-01\\
  1.20e-01 & -1.61e-02 & 7.32e-01 & -2.33e-01 & -4.15e-02 & 4.28e-01 & -4.82e-01 & 1.58e-01 & -2.46e-01 & 7.98e-03 & 7.01e-03 & -1.34e-01 & -6.39e-01 & -5.28e-01 & -5.45e-01 & -6.89e-01 & 1.13e-03 & 3.71e-02 & 3.06e-01 & -7.26e-02 & 4.91e-01 & -1.05e+00 & -5.16e-01 & -1.13e-02 & 
-2.37e-01 & 4.80e-01 & -8.23e-01 & 4.18e-01 & -2.84e-01 & 1.66e-01\\
  -5.30e-01 & -3.81e-01 & 8.08e-02 & -2.41e-01 & -3.75e-03 & 1.44e-01 & 9.89e-01 & -1.77e-01 & 2.47e-01 & 2.54e-01 & 8.17e-02 & 2.34e-02 & -1.02e-01 & 1.13e-02 & 2.17e-01 & 4.97e-01 & 2.24e-01 & 2.41e-01 & -1.25e-01 & 1.43e-01 & 3.18e-01 & 5.69e-02 & -1.11e-03 & -2.78e-01 & -1.00e-01 & -2.23e-02 & -3.20e-01 & -1.21e-02 & -4.73e-01 & -2.14e-01\\
  -3.67e-01 & 2.17e-01 & 1.63e-01 & 4.74e-01 & 3.85e-02 & 8.19e-01 & 9.11e-01 & -3.28e-01 & -4.70e-02 & -1.69e-01 & 2.76e-01 & 3.70e-01 & -2.23e-01 & 4.52e-01 & -1.17e-01 & -1.01e-01 & 5.21e-01 & 5.43e-01 & 1.28e-01 & 3.47e-01 & -2.42e-01 & -2.80e-01 & -4.14e-01 & -2.96e-01 & 6.62e-01 & -1.57e-02 & -4.29e-01 & 5.55e-03 & -3.30e-02 & 1.92e-01\\
  5.57e-02 & -2.61e-01 & -4.05e-02 & -1.09e-01 & -4.40e-01 & 3.29e-01 & 1.78e-01 & -1.14e-01 & 4.38e-01 & -1.39e-01 & -3.94e-01 & 2.67e-01 & 3.91e-01 & -5.33e-01 & 1.46e-01 & 3.57e-01 & 2.39e-01 & -1.65e-01 & -3.99e-01 & 1.38e-01 & 3.57e-01 & 3.68e-01 & -3.90e-01 & -6.30e-02 & 5.36e-01 & 3.88e-01 & 4.94e-01 & -7.19e-01 & 5.54e-01 & -6.84e-01\\
  -3.23e-01 & -1.15e-01 & -1.55e-01 & -1.72e-02 & -5.97e-02 & -3.24e-01 & 1.63e-01 & 9.84e-02 & 3.68e-01 & -2.94e-01 & 1.95e-01 & 2.13e-01 & -6.06e-01 & 5.30e-01 & 5.27e-01 & 6.23e-01 & -5.19e-01 & -1.68e-01 & 6.22e-03 & -9.78e-02 & 5.50e-01 & -6.58e-01 & 9.16e-02 & -2.50e-01 & 2.83e-01 & -2.99e-01 & 2.85e-02 & 3.08e-01 & 1.08e-01 & -3.88e-01\\
  5.85e-02 & -6.09e-02 & 2.19e-01 & -2.63e-01 & -1.41e-01 & -4.13e-01 & 1.22e+00 & 3.28e-01 & -6.35e-01 & 5.26e-02 & 2.12e-02 & 1.07e-01 & -3.58e-01 & -1.60e-01 & -7.41e-02 & 2.15e-01 & 6.47e-02 & 3.47e-01 & -3.94e-01 & 2.25e-01 & -5.62e-02 & 2.99e-01 & 2.51e-01 & -4.11e-02 & 2.50e-01 & 5.26e-01 & -4.88e-02 & -1.05e-01 & -4.70e-02 & 4.49e-01\\
  -2.03e-01 & 4.43e-02 & -2.60e-01 & -1.05e-01 & -1.99e-01 & 5.13e-01 & 6.73e-01 & -8.13e-02 & -4.36e-01 & 9.82e-02 & -6.53e-02 & 4.43e-01 & 2.28e-01 & -3.77e-01 & -3.92e-01 & 6.21e-02 & 1.30e-01 & -4.44e-01 & -1.84e-01 & -4.44e-01 & 3.90e-01 & -7.02e-02 & 1.67e-01 & -2.47e-01 & 
5.37e-01 & 2.62e-01 & 4.60e-01 & -1.12e-01 & 3.32e-01 & 2.23e-02\\
  2.38e-01 & 4.11e-01 & -2.64e-01 & -6.30e-02 & -3.45e-01 & -4.68e-01 & -3.79e-01 & -3.82e-01 & 6.56e-02 & -1.87e-02 & 3.30e-01 & -1.31e-02 & -4.36e-01 & 6.77e-01 & -6.43e-02 & -5.39e-01 & 5.22e-01 & 2.36e-01 & -4.58e-01 & 3.15e-01 & 7.15e-01 & -6.03e-01 & -6.56e-01 & 2.43e-01 & 
-4.38e-01 & 3.60e-01 & -1.20e-01 & -1.99e-01 & 6.68e-01 & -3.79e-01\\
  8.77e-02 & 1.48e-02 & -3.90e-02 & 8.91e-02 & 8.81e-04 & 4.73e-02 & 1.80e-01 & 1.51e-02 & 5.51e-01 & -5.89e-01 & 3.66e-01 & 7.79e-01 & 6.58e-02 & 2.14e-03 & -2.46e-01 & -1.87e-01 & -1.01e-01 & -6.43e-02 & -6.09e-02 & -1.85e-01 & 8.33e-01 & -8.58e-02 & 1.48e-01 & -8.72e-01 & -2.77e-02 & -3.32e-01 & 6.29e-01 & -3.34e-01 & 7.44e-02 & -2.61e-01\\
  -6.96e-01 & 4.06e-01 & 1.31e-01 & 4.42e-01 & 2.47e-01 & 3.46e-01 & 2.12e-01 & 5.11e-01 & 2.61e-01 & -2.68e-01 & 8.43e-01 & 5.09e-01 & 3.69e-01 & 1.57e-01 & 6.78e-01 & 6.22e-02 & 3.38e-01 & 2.33e-01 & 4.28e-02 & -4.84e-01 & 7.41e-01 & 8.94e-01 & 2.48e-01 & -8.71e-02 & 4.59e-01 & 1.72e-02 & 3.16e-01 & 2.55e-01 & -1.89e-02 & 8.70e-02\\
  -9.29e-01 & 2.55e-01 & 3.58e-01 & 4.56e-01 & 2.61e-01 & 1.14e-01 & -7.44e-01 & 3.80e-02 & -4.97e-02 & -8.95e-02 & -7.87e-02 & 8.46e-01 & -1.52e-02 & -1.95e-01 & -5.42e-01 & -7.93e-01 & 9.05e-02 & 3.40e-01 & 4.54e-01 & -1.59e-01 & -2.06e-01 & -4.05e-01 & -8.69e-01 & 2.09e-01 & -4.64e-01 & -4.76e-01 & -4.98e-01 & 7.74e-02 & 1.25e-01 & 7.94e-01\\
  -2.56e-01 & -5.06e-01 & 1.34e-01 & -3.52e-01 & 1.51e-01 & -1.56e-01 & 4.59e-01 & 3.42e-03 & 3.96e-01 & 4.48e-02 & 5.63e-01 & -1.41e-01 & 4.18e-01 & -1.67e-01 & 2.70e-01 & -2.38e-01 & 4.16e-01 & 1.60e-01 & 1.44e-01 & 1.70e-01 & -2.39e-02 & -2.85e-01 & 3.68e-01 & 6.08e-01 & 6.26e-01 & 4.44e-01 & 3.12e-01 & -1.95e-01 & 3.12e-01 & -1.77e-01\\
  6.27e-01 & -2.40e-01 & -3.48e-01 & -2.89e-01 & -1.77e-01 & 4.07e-01 & 7.71e-01 & 8.90e-01 & 7.73e-01 & 2.66e-01 & 2.77e-01 & -2.84e-03 & 3.92e-01 & 2.96e-01 & -7.04e-01 & -7.29e-02 & 4.64e-01 & -3.75e-01 & -1.23e-01 & 3.35e-01 & 8.18e-01 & 1.08e-01 & 3.00e-01 & 4.25e-02 & 2.47e-01 & -4.22e-02 & 1.03e-01 & 6.32e-02 & -4.33e-01 & -1.43e-02\\
  -1.28e-01 & 1.67e-02 & -4.86e-01 & -1.98e-01 & 1.44e-01 & 8.93e-02 & -3.36e-01 & 9.12e-02 & 7.35e-01 & 2.57e-01 & -1.66e-01 & 1.79e-01 & -1.43e-01 & -1.46e-01 & -1.47e-01 & -1.81e-01 & 7.83e-01 & -2.71e-01 & -2.75e-01 & -4.07e-01 & -3.66e-01 & -6.67e-01 & -3.26e-01 & 1.14e-01 & 1.01e-01 & -1.64e-01 & -1.08e+00 & 6.94e-01 & 2.54e-01 & 3.40e-01\\
  -1.56e-01 & -4.08e-01 & -2.23e-01 & -9.35e-02 & 1.40e-01 & 8.29e-01 & 1.38e+00 & -2.80e-01 & -1.21e+00 & 3.40e-01 & -2.36e-01 & -4.58e-01 & -1.65e-01 & 6.90e-01 & 4.32e-01 & -3.00e-01 & 1.04e-01 & 3.11e-01 & -3.02e-01 & 1.12e-01 & 1.82e-01 & -2.67e-01 & -3.87e-01 & -1.37e-01 & 
4.76e-01 & -3.97e-02 & -7.04e-02 & 2.32e-01 & -2.08e-01 & 2.06e-01\\
  -2.08e-01 & -6.29e-01 & 1.39e-01 & -1.78e-01 & 1.55e-01 & 1.04e-01 & -1.23e-02 & -4.80e-01 & -2.59e-01 & -9.66e-03 & -1.17e-01 & 1.14e-01 & -5.67e-02 & -3.48e-01 & -9.73e-02 & 3.29e-01 & 1.22e-01 & 5.57e-03 & -4.31e-02 & 9.05e-02 & -1.65e-01 & 3.89e-02 & -5.23e-01 & 3.17e-01 & 
6.08e-02 & 3.49e-01 & -6.30e-01 & 3.97e-01 & 8.65e-02 & 2.25e-02\\
  5.16e-01 & -3.15e-01 & 3.57e-01 & 1.73e-01 & -1.14e-01 & -2.42e-01 & -5.76e-02 & -1.78e-01 & -7.81e-02 & -2.09e-01 & -4.03e-01 & 5.90e-01 & -1.89e-01 & -2.62e-01 & 2.42e-01 & -5.95e-03 & -9.70e-03 & -3.19e-01 & 1.80e-01 & -2.60e-01 & -2.95e-01 & 1.96e-01 & -7.49e-01 & 9.50e-02 
& -1.91e-01 & -6.89e-01 & 1.30e-01 & -5.94e-02 & 6.21e-01 & 8.76e-02\\
  7.45e-02 & 4.42e-01 & -1.16e-01 & -6.98e-01 & 6.59e-02 & 1.81e-01 & 4.16e-01 & 3.63e-02 & 1.98e-01 & -6.84e-02 & -3.70e-01 & -9.90e-02 & -2.46e-01 & -5.26e-01 & 4.10e-01 & 1.60e-01 & 3.08e-01 & -3.90e-01 & 2.31e-01 & -5.02e-01 & 1.02e-02 & -2.39e-01 & -9.00e-02 & -2.88e-01 & -1.93e-01 & -3.38e-01 & 6.02e-01 & -2.27e-01 & 1.14e+00 & 5.18e-01\\
  6.25e-01 & -3.84e-03 & -4.61e-01 & -3.28e-01 & -1.16e-01 & 3.63e-01 & 9.07e-02 & 8.26e-01 & -3.77e-01 & 2.65e-01 & -1.44e-01 & 2.83e-01 & -2.74e-01 & 6.19e-01 & 5.21e-01 & 3.52e-01 & -1.83e-01 & 4.18e-01 & -2.34e-02 & -5.13e-02 & -3.56e-01 & -1.73e-01 & 3.14e-01 & 1.20e-01 & -3.19e-01 & -1.06e-01 & 7.07e-02 & -3.32e-01 & 5.54e-01 & 6.50e-01\\
  -1.01e-01 & -3.86e-01 & 3.27e-01 & 2.11e-01 & 5.43e-01 & 9.87e-01 & 6.66e-01 & 8.07e-01 & -5.60e-01 & -2.60e-01 & -2.05e-01 & 5.48e-03 & 3.39e-01 & -4.47e-01 & 1.04e-01 & -2.63e-01 & -6.93e-01 & 7.47e-01 & 6.84e-01 & -2.77e-01 & 2.42e-02 & -3.88e-01 & 6.72e-02 & -4.45e-01 & 4.72e-01 & 1.82e-01 & -5.03e-01 & 2.62e-01 & -2.86e-01 & 1.96e-01\\
  -3.18e-01 & -3.09e-01 & 2.80e-01 & -4.85e-01 & -2.03e-02 & 5.75e-01 & 7.36e-01 & -5.28e-01 & 3.78e-01 & 8.03e-01 & 2.40e-01 & -3.58e-01 & -2.81e-01 & 1.06e-01 & 3.86e-01 & 6.97e-02 & -3.79e-01 & 5.06e-01 & 4.28e-02 & -2.66e-01 & 2.47e-01 & 2.02e-01 & -3.10e-01 & 5.93e-01 & 4.43e-01 & 5.41e-02 & 1.48e-01 & -5.71e-02 & -4.77e-02 & -1.65e-01\\
  3.09e-02 & 6.74e-01 & 3.09e-01 & 4.22e-01 & 1.78e-01 & 2.28e+00 & 8.14e-01 & 6.22e+00 & 5.79e+00 & 1.96e+00 & 3.73e-02 & -2.50e-02 & 1.83e-01 & -2.14e-01 & -3.33e-01 & 2.07e-01 & 2.74e-01 & 1.12e-01 & -2.62e-01 & -2.21e-01 & 2.00e-01 & -1.23e-01 & 4.32e-01 & -1.65e-01 & 6.30e-01 & -5.33e-02 & 1.05e-01 & -5.64e-02 & 1.31e-01 & 5.15e-01\\
  1.12e-01 & 5.49e-02 & 7.04e-01 & -5.05e-01 & -3.83e-01 & -6.66e-01 & 2.11e-02 & 1.07e+00 & -1.51e+00 & 3.01e-01 & 1.08e-01 & -2.15e-01 & -4.04e-01 & 3.26e-02 & -1.45e-01 & -3.39e-01 & 2.10e-01 & -3.16e-01 & 2.93e-02 & 1.30e-01 & 4.83e-01 & -5.33e-01 & -4.50e-01 & 2.64e-01 & -3.84e-03 & 6.61e-02 & 5.40e-03 & 1.96e-01 & 9.46e-02 & -1.15e-01\\
  8.37e-02 & 7.58e-01 & 3.04e-01 & 3.82e-01 & 1.69e-02 & 6.62e-01 & 2.58e-01 & 4.95e+00 & -8.76e-01 & -1.06e-01 & -5.23e-01 & -2.20e-01 & -1.09e-01 & -4.29e-02 & -1.26e-02 & -2.74e-01 & 1.66e-01 & -9.05e-02 & 4.30e-01 & -6.20e-02 & 7.92e-02 & 5.01e-01 & -5.37e-03 & 5.10e-02 & 7.38e-01 & -2.53e-01 & 1.99e-02 & 5.13e-01 & 1.88e-02 & -4.76e-02\\
  9.80e-03 & -3.09e-01 & 2.00e-02 & -5.40e-01 & -3.04e-01 & 5.17e-02 & -8.23e-02 & -1.02e-01 & -1.08e+00 & 3.05e-01 & 5.08e-01 & 4.36e-01 & 1.33e-01 & -7.35e-01 & 4.16e-01 & 3.87e-01 & 3.50e-01 & 7.52e-01 & 3.75e-01 & 2.13e-01 & 1.59e-01 & 5.28e-01 & -1.77e-01 & 1.34e-01 & 2.17e-01 & 3.16e-01 & 5.96e-01 & -6.29e-01 & -1.89e-01 & 3.65e-02\\
  -2.90e-01 & -1.15e+00 & 1.14e-01 & 1.38e-01 & 4.58e-01 & -3.34e-01 & -8.01e-02 & -1.71e-01 & 3.39e-01 & -5.67e-02 & 1.13e-01 & -8.59e-01 & -1.85e-01 & 3.62e-01 & 3.34e-01 & 5.29e-01 & 8.08e-02 & 1.37e-01 & -3.83e-02 & 6.51e-02 & -6.25e-02 & -2.36e-01 & 1.35e-01 & -1.99e-01 & -1.66e-01 & -4.40e-01 & 5.36e-01 & 2.91e-01 & -1.58e-01 & 3.59e-01\\
  -2.32e-01 & -1.38e-01 & -2.29e-01 & 5.84e-01 & -4.02e-02 & 7.44e-01 & 1.17e-01 & 7.39e-01 & -5.79e-01 & 8.78e-02 & 4.95e-02 & -5.74e-02 & 8.12e-02 & 3.11e-01 & -8.32e-02 & 4.68e-01 & 5.75e-01 & 1.07e-01 & -3.45e-01 & -3.60e-01 & -1.05e+00 & 2.40e-01 & 5.73e-01 & 2.96e-01 & 6.61e-02 & -1.24e-01 & -7.89e-01 & -5.19e-02 & -2.13e-01 & 2.72e-01\\
  -4.83e-01 & -3.51e-01 & -5.95e-01 & 1.83e-02 & 1.62e-01 & 2.69e+00 & 2.00e+00 & -1.17e+00 & -4.59e+00 & 7.58e-01 & -2.32e-01 & 5.27e-01 & -2.56e-01 & -2.83e-01 & 3.82e-01 & -1.66e-01 & -7.12e-02 & -1.75e-01 & 1.47e-01 & 1.66e-02 & -4.15e-01 & 1.16e+00 & -4.29e-02 & -9.52e-02 & 
1.32e-01 & -1.06e-01 & 2.69e-01 & 9.23e-02 & -8.45e-02 & 5.84e-01\\
  -4.29e-02 & -2.28e-01 & 5.28e-03 & 2.94e-01 & 1.48e-02 & 5.28e-01 & 1.31e+00 & -1.33e+00 & -5.99e-01 & 6.79e-01 & -3.70e-01 & 3.13e-01 & -2.04e-01 & -2.98e-02 & 6.84e-02 & 3.74e-01 & 1.16e-01 & -4.55e-02 & 2.69e-01 & 2.17e-02 & -1.11e-01 & 3.08e-01 & -2.93e-01 & 9.52e-02 & 1.69e-01 & 3.91e-01 & -2.77e-01 & 1.97e-01 & -5.55e-02 & 1.20e-01\\
  3.80e-01 & -9.89e-02 & -1.17e-01 & 6.35e-01 & 1.41e-02 & 1.20e+00 & 7.37e-01 & 1.06e+00 & -2.84e-01 & 2.41e-01 & -2.61e-01 & 3.17e-01 & 1.12e-01 & -1.17e-02 & 4.74e-01 & -2.74e-01 & -1.70e-01 & 4.16e-02 & 1.86e-01 & -4.30e-02 & 2.19e-01 & -2.78e-01 & 1.71e-01 & 2.26e-01 & 1.66e-01 & -1.49e-02 & 1.60e-01 & 4.98e-02 & 9.14e-01 & -1.87e-01\\
  -7.37e-02 & -1.86e-01 & -8.15e-01 & 1.30e-02 & -1.38e-01 & -4.46e-01 & -4.13e-02 & -2.60e-01 & 3.65e-01 & 6.63e-02 & -8.08e-02 & 4.02e-02 & -3.23e-01 & -2.52e-01 & 2.16e-01 & 4.42e-01 & -4.03e-01 & 1.47e-01 & 8.12e-01 & -1.69e-01 & -2.96e-01 & -6.52e-01 & -3.51e-01 & -1.73e-01 
& 2.57e-01 & -9.40e-02 & -4.60e-01 & -5.87e-01 & 6.92e-02 & 2.91e-01\\
  -2.64e-01 & 1.57e-01 & -1.41e-01 & -2.98e-01 & 1.59e-01 & 1.73e-01 & 2.92e-01 & 1.96e-01 & 1.71e-01 & 1.25e-01 & 3.35e-01 & 3.10e-01 & -9.56e-02 & -2.56e-01 & -5.47e-01 & 5.63e-01 & -3.79e-01 & 3.68e-01 & 5.63e-01 & -2.11e-01 & 1.19e-01 & 7.37e-01 & 3.46e-02 & -6.11e-01 & -4.18e-01 & -2.89e-01 & -3.58e-01 & -2.82e-01 & -3.28e-02 & -1.80e-01\\
  2.27e-02 & 5.60e-01 & 4.47e-02 & 1.04e-01 & 4.95e-02 & -8.95e-02 & -5.82e-02 & 5.48e-01 & -3.86e-01 & -2.31e-02 & 2.42e-01 & -6.30e-02 & -5.87e-01 & 2.14e-01 & 6.69e-02 & -2.79e-02 & 1.05e+00 & 8.56e-02 & 4.01e-01 & 4.06e-01 & -3.60e-02 & 2.18e-01 & -1.33e-01 & 1.10e-01 & 7.85e-01 & 2.77e-01 & -2.26e-01 & 5.08e-02 & 2.01e-01 & 2.19e-02\\
  -3.77e-02 & -6.89e-02 & 2.23e-01 & 3.78e-01 & -1.72e-01 & -2.31e-01 & -2.47e-01 & 5.07e-01 & 4.26e-01 & 2.16e-01 & 3.25e-01 & 2.73e-01 & 4.96e-02 & -4.31e-01 & -2.80e-01 & -2.27e-01 & 2.77e-02 & -4.57e-01 & -2.97e-01 & 2.01e-01 & 7.69e-02 & 2.85e-01 & -1.41e-01 & -4.38e-01 & -2.40e-01 & 7.84e-02 & 1.37e-01 & -2.26e-01 & 1.51e-01 & 1.91e-01\\
  -2.56e-01 & -3.22e-01 & -6.39e-02 & -5.69e-01 & 3.16e-01 & 1.09e+00 & 6.00e-01 & -1.57e-01 & -4.19e-01 & 4.71e-01 & -1.33e-01 & -3.61e-01 & 1.15e-01 & -1.33e-01 & 2.11e-01 & 2.20e-01 & 5.78e-02 & -3.07e-01 & 9.59e-01 & 5.86e-01 & -1.57e-01 & 7.29e-02 & -8.66e-01 & -5.54e-02 & 4.60e-01 & 5.59e-01 & 2.72e-01 & -1.89e-01 & 3.44e-01 & 9.72e-02\\
  -1.61e-01 & -2.75e-01 & -1.08e-01 & 3.97e-01 & 7.07e-02 & 2.64e-01 & 4.07e-01 & 2.35e+00 & 6.81e-01 & 4.41e-01 & 5.12e-01 & 1.70e-01 & -1.61e-01 & -6.80e-01 & -1.33e-01 & 1.15e-01 & 2.58e-01 & -3.91e-01 & 5.25e-03 & 3.27e-01 & -1.49e-01 & -2.83e-02 & -1.96e-01 & -3.57e-01 & 1.36e-01 & -4.08e-01 & 5.14e-02 & -4.30e-02 & 2.80e-01 & -1.03e-01\\
  2.33e-01 & -3.94e-01 & -6.62e-02 & -2.21e-02 & -4.80e-02 & 1.30e+00 & 3.67e-01 & -1.01e+00 & -1.28e+00 & 1.88e-01 & 1.69e-01 & 2.79e-01 & 3.87e-01 & 1.17e-01 & 6.14e-02 & -2.04e-01 & 1.98e-01 & 6.00e-01 & 5.84e-01 & 3.95e-01 & 1.14e-01 & 7.07e-01 & -4.92e-01 & -1.70e-01 & -4.06e-01 & -3.53e-01 & 6.10e-02 & 2.58e-01 & 7.26e-01 & 3.71e-01\\
  6.70e-02 & -1.62e-02 & -2.81e-01 & 4.87e-02 & -2.21e-01 & 9.00e-01 & 6.42e-02 & 4.47e-01 & 2.88e-01 & 3.25e-01 & -3.16e-01 & -3.13e-01 & 1.65e-01 & -3.79e-01 & 2.36e-01 & -1.18e-01 & 7.68e-01 & -2.57e-03 & 3.95e-01 & 1.39e-01 & -1.33e-01 & 2.94e-01 & -2.52e-01 & 7.44e-01 & 3.78e-01 & 5.29e-01 & 1.10e-01 & -2.71e-01 & 2.68e-01 & -6.39e-01\\
  -2.58e-01 & -1.32e-01 & -1.31e-01 & 5.67e-01 & -2.10e-02 & -1.02e-01 & -3.03e-01 & 4.84e-01 & 2.87e-01 & 1.03e-01 & 1.32e-02 & 5.86e-01 & -3.84e-01 & -5.18e-01 & -4.77e-02 & -1.15e-01 & 3.91e-01 & 3.18e-01 & 1.68e-01 & -3.54e-01 & -1.36e-01 & -5.01e-01 & 1.66e-01 & -2.09e-01 & 
-2.68e-01 & 3.78e-01 & -7.81e-01 & -2.12e-01 & 3.71e-01 & 4.54e-01\\
  -1.57e-01 & -1.36e-01 & 5.43e-01 & 1.75e-02 & 1.42e-01 & -4.82e-01 & 2.25e-01 & -9.01e-01 & -1.72e-01 & 1.34e-01 & 2.17e-01 & 4.89e-01 & -2.64e-01 & 1.68e-01 & 5.73e-01 & -8.13e-01 & 3.01e-01 & 6.60e-01 & 3.08e-01 & -3.28e-02 & -1.67e-01 & -1.80e-01 & -3.82e-01 & -1.09e-01 & 3.81e-01 & 2.44e-01 & -3.38e-01 & 5.56e-01 & 1.16e-01 & -8.41e-01\\
  1.58e-01 & 1.03e-01 & 3.70e-01 & 2.46e-02 & -1.05e-01 & -2.26e-01 & 1.04e-01 & 2.95e-02 & 2.74e-01 & -1.10e-01 & -5.24e-01 & 3.16e-01 & 2.02e-01 & -1.59e-01 & 1.71e-02 & 2.41e-01 & 2.23e-01 & 8.65e-02 & -1.15e-01 & 5.23e-02 & 9.11e-01 & 3.42e-01 & 1.66e-01 & 2.88e-01 & 2.32e-01 & -4.50e-01 & 6.82e-02 & -9.58e-01 & 7.40e-01 & -2.03e-01\\
  1.81e-01 & -2.18e-03 & -4.51e-02 & 2.12e-01 & -1.94e-01 & 4.86e-01 & -1.13e-01 & -4.38e-01 & -1.40e-01 & -1.73e-01 & 2.77e-01 & -4.17e-01 & -4.47e-02 & -3.93e-01 & -3.07e-01 & 5.63e-01 & -3.99e-01 & 1.84e-01 & 3.73e-01 & 5.09e-01 & -2.24e-01 & -3.48e-01 & -2.01e-01 & -6.23e-01 
& -2.53e-02 & -4.63e-01 & 4.55e-01 & 4.21e-01 & 6.28e-01 & 1.74e-01\\
  -1.80e-01 & 1.26e-01 & 3.13e-01 & 1.86e-01 & -6.77e-02 & -8.86e-03 & 7.74e-01 & -5.56e-01 & 2.96e-01 & 2.69e-01 & 2.71e-01 & -6.89e-02 & -5.54e-01 & 3.18e-02 & 3.02e-02 & 9.49e-02 & 2.60e-02 & -4.48e-01 & -1.14e-01 & 3.97e-01 & 2.65e-01 & 1.09e+00 & -1.51e-02 & 2.11e-01 & 1.55e-01 & -4.49e-01 & -2.53e-02 & -2.47e-02 & 3.31e-01 & -4.60e-01\\
  5.80e-02 & -2.94e-01 & -3.81e-01 & 6.12e-01 & -5.01e-01 & 2.08e-02 & -6.30e-01 & 2.95e-01 & 2.12e-01 & -2.30e-01 & -3.72e-01 & -6.87e-01 & 6.47e-01 & 1.18e-01 & 4.34e-01 & 7.32e-01 & -9.67e-02 & -1.59e-01 & -3.40e-01 & -2.29e-01 & -8.15e-02 & 4.07e-01 & -1.73e-01 & 6.60e-01 & 8.83e-01 & -4.92e-01 & -1.42e-01 & -1.62e-01 & 6.72e-01 & -2.97e-01\\
  -1.54e-01 & 3.57e-01 & -1.21e-01 & 5.42e-02 & -5.69e-01 & 2.50e-01 & 5.41e-01 & 5.46e-01 & 1.67e-01 & 1.00e-02 & 3.74e-01 & 1.29e-01 & -4.48e-01 & 2.88e-01 & -1.73e-01 & 3.32e-01 & -2.43e-01 & -6.57e-01 & 4.27e-01 & -3.28e-01 & -3.69e-01 & -6.90e-02 & -2.03e-01 & 3.57e-02 & -5.55e-01 & 1.20e-01 & 6.69e-01 & -1.72e-01 & 2.95e-01 & 1.89e-02\\
  6.20e-01 & 4.95e-01 & -3.67e-01 & -2.08e-01 & 1.66e-01 & 7.65e-02 & -3.16e-02 & -7.81e-02 & -6.56e-02 & 1.06e-01 & -6.84e-01 & 2.23e-02 & 1.61e-01 & 1.51e-01 & 5.47e-01 & 1.10e+00 & 3.96e-02 & 9.58e-02 & 2.65e-02 & -4.44e-01 & -5.71e-01 & 3.88e-01 & 3.12e-02 & 3.23e-02 & -9.58e-02 & 5.94e-01 & -4.39e-01 & 3.60e-02 & 1.78e-01 & -1.53e-01\\
\end{bmatrix}
$$

1st image in the batch has objects in row 24 and 30 (human, dog at 3,3 and 2,4)

- loop over batch
    - `y_true` is the ground truth matrix of shape `(49, 30)`;
    - `y_pred` is the predicted matrix of shape `(49, 30)`;
    - loop over grid cells from 0 to 48
    - loop 0 `i=0`:
        - `y_true_i` is the ground truth matrix of shape `(1, 30)` for the cell `i=0`;
        - Notice that we know in advance that in this grid cell $0$ there is no ground truth object
          and hence `y_true_i` is an all zero vector, by {prf:ref}`yolo_gt_matrix`.

        - `y_pred_i` is the predicted matrix of shape `(1, 30)` for the cell `i=0`;
        
        - `indicator_obj_i` corresponds to $\obji$ in the equation above and is
          equal to $0$ since there is no object in cell $i=0$. 
        
        - We do not explicitly define `indicator_noobj_i` to correspond to $\nobji$
          but it is equal to $1$ since there is no object in cell $i=0$. This means
          that it will not go through the `if` clause in `line ??` and will go through
          `else` clause in `line ??` and will compute the loss for the no object equation
          at $d$ equation above.

        - **No object loss**: `line xx-xx` means we are looping over the 2 bounding boxes in cell $i=0$.
            - Loop over $j=0$
                - `y_true[i, 4]` is the confidence score of the 1st bounding box in cell $i=0$.
                - It is equal to $0$ since there is no object in cell $i=0$ by {prf:ref}`gt-confidence`.
                - `y_pred[i, 4]` is the confidence score of the 1st bounding box in cell $i=0$.
                - We will compute the mean squared error between the ground truth confidence
                score and the predicted confidence score of the 1st predicted bounding box.
            - Loop over $j=1$
                - We still use `y_true[i, 4]` because the `y_true[i, 9]` is same as `y_true[i, 4]` by
                  construction in {prf:ref}`yolo_gt_matrix`.  
                - `y_pred[i, 9]` is the confidence score of the 2nd bounding box in cell $i=0$.
                - We will compute the mean squared error between the ground truth confidence score and
                  the predicted confidence score of the 2nd predicted bounding box.
            - Finally, these two errors are summed up to be `self.no_object_conf_loss`. We will put the 
              $\lambda_\textbf{noobj}$ in front of this loss in the `line ??` of the code later.


## References

s



[^1]: https://www.harrysprojects.com/articles/yolov1.html