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
$$

# YOLOv1

## Notations and Definitions

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
batch_size = 16
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
$$

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

### Bipartite Matching

pass


[^1]: https://www.harrysprojects.com/articles/yolov1.html