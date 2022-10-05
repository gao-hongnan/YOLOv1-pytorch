"""
Implementation of Yolo Loss Function from the original yolo paper
"""
import torch
from torch import nn
from utils import intersection_over_union, bmatrix
import numpy as np
import pandas as pd
# reshape to [49, 30] from [7, 7, 30]
class YOLOv1Loss2D(nn.Module):
    # assumptions
    # 1. B = 2 is a must since we have iou_1 vs iou_2
    def __init__(
        self,
        S: int,
        B: int,
        C: int,
        lambda_coord: float = 5,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # mse = (y_pred - y_true)^2
        # FIXME: still unclean cause by right reduction is not sum since we are
        # adding scalars, but class_loss uses vector sum reduction so need to use for all?
        self.mse = nn.MSELoss(reduction="none")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # important step if not the curr batch loss will be added to the next batch loss which causes error.
    def _initiate_loss(self) -> None:
        # bbox loss
        self.bbox_xy_offset_loss = 0
        self.bbox_wh_loss = 0

        # objectness loss
        self.object_conf_loss = 0
        self.no_object_conf_loss = 0

        # class loss
        self.class_loss = 0

    def compute_xy_offset_loss(self, x_i, xhat_i, y_i, yhat_i) -> torch.Tensor:
        return self.mse(x_i, xhat_i) + self.mse(y_i, yhat_i)

    def compute_wh_loss(
        self, w_i, what_i, h_i, hhat_i, epsilon: float = 1e-6
    ) -> torch.Tensor:
        return self.mse(
            torch.sqrt(w_i), torch.sqrt(torch.abs(what_i + epsilon))
        ) + self.mse(torch.sqrt(h_i), torch.sqrt(torch.abs(hhat_i + epsilon)))

    def compute_object_conf_loss(self, conf_i, confhat_i) -> torch.Tensor:
        return self.mse(conf_i, confhat_i)

    def compute_no_object_conf_loss(self, conf_i, confhat_i) -> torch.Tensor:
        return self.mse(conf_i, confhat_i)

    def compute_class_loss(self, p_i, phat_i) -> torch.Tensor:
        # self.mse(p_i, phat_i) is vectorized version
        total_class_loss = 0
        for c in range(self.C):
            total_class_loss += self.mse(p_i[c], phat_i[c])
        return total_class_loss

    def forward(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
        self._initiate_loss()
        print("y_trues.shape", y_trues.shape)
        print("y_preds.shape", y_preds.shape)

        # y_trues: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
        # y_preds: (batch_size, S, S, C + B * 5) = (batch_size, 7, 7, 30) -> (batch_size, 49, 30)
        y_trues = y_trues.reshape(-1, self.S * self.S, self.C + self.B * 5)
        y_preds = y_preds.reshape(-1, self.S * self.S, self.C + self.B * 5)

        batch_size = y_preds.shape[0]

        # batch_index is the index of the image in the batch
        for batch_index in range(batch_size):
            # print(f"batch_index {batch_index}")

            # this is the gt and pred matrix in my notes: y and yhat
            y_true = y_trues[batch_index]  # (49, 30)
            y_pred = y_preds[batch_index]  # (49, 30)
            
            # print(bmatrix(y_true.cpu().detach().numpy()))
            # print(bmatrix(y_pred.cpu().detach().numpy()))
            y_true_col_names = ["x", "y", "w", "h", "conf", "x", "y", "w", "h", "conf", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"]
            y_pred_col_names = ["xhat^1", "yhat^1", "what^1", "hhat^1", "confhat^1", "xhat^2", "yhat^2", "what^2", "hhat^2", "confhat^2", "p1hat", "p2hat", "p3hat", "p4hat", "p5hat", "p6hat", "p7hat", "p8hat", "p9hat", "p10hat", "p11hat", "p12hat", "p13hat", "p14hat", "p15hat", "p16hat", "p17hat", "p18hat", "p19hat", "p20hat"]
            y_true_df = pd.DataFrame(data=y_true.cpu().detach().numpy(), index =[i for i in range(self.S * self.S)], columns = y_true_col_names)
            y_true_df.to_csv("y_true.csv")
            y_pred_df = pd.DataFrame(data=y_pred.cpu().detach().numpy(), index =[i for i in range(self.S * self.S)], columns = y_pred_col_names)
            y_pred_df.to_csv("y_pred.csv")
            print(y_true_df.to_markdown())
            
            # print(y_true_df.head())
            # print(f"y_true {y_true}")

            # i is the grid cell index in my notes ranging from 0 to 48
            for i in range(self.S * self.S):
                # print(f"row/grid cell index {i}")

                y_true_i = y_true[i]  # (30,) or (1, 30)
                y_pred_i = y_pred[i]  # (30,) or (1, 30)

                # print(f"y_i={y_true_i}")
                # print(f"yhat_i={y_pred_i}")

                # b = y_true_i[0:4]

                # this is $\obji$ and checking y_true[i, 4] is sufficient since y_true[i, 9] is repeated
                indicator_obj_i = y_true_i[4] == 1

                # here equation 1, 2, 3, 5
                if indicator_obj_i:

                    b = y_true_i[0:4]
                    bhat_1 = y_pred_i[0:4]
                    bhat_2 = y_pred_i[5:9]
                    # print(f"b {b}\nbhat_1 {bhat_1}\nbhat_2 {bhat_2}")

                    x_i, y_i, w_i, h_i = b
                    # print(f"x_i {x_i}, y_i {y_i}, w_i {w_i}, h_i {h_i}")

                    p_i = y_true_i[10:]
                    phat_i = y_pred_i[10:]

                    # area of overlap
                    iou_b1 = intersection_over_union(b, bhat_1, bbox_format="yolo")
                    iou_b2 = intersection_over_union(b, bhat_2, bbox_format="yolo")

                    # max iou
                    if iou_b1 > iou_b2:
                        xhat_i, yhat_i, what_i, hhat_i = bhat_1
                        # conf_i = max_{bhat \in {bhat_1, bhat_2}} IoU(b, bhat)
                        # however I set conf_i = y_true_i[4] = 1 here as it gives better results for our case
                        conf_i, confhat_i = y_true_i[4], y_pred_i[4]
                        
                        # fmt: off
                        # equation 1
                        self.bbox_xy_offset_loss += self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i, y_i, yhat_i)

                        # make them abs as sometimes the preds can be negative if no sigmoid layer.
                        # add 1e-6 for stability
                        self.bbox_wh_loss += self.lambda_coord * self.compute_wh_loss(w_i, what_i, h_i, hhat_i)

                        self.object_conf_loss += self.compute_object_conf_loss(conf_i, confhat_i)

                        # mention 2 other ways
                        # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                        # we can set conf_i = iou_b2 as well as the smaller of the two should be optimized to say there exist no object instead of proposing something.
                        # we can set conf_i = 0 as well and it will work.
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=torch.tensor(0., device="cuda"), confhat_i=y_pred_i[9])
                    else:
                        xhat_i, yhat_i, what_i, hhat_i = bhat_2
                        # same as above
                        conf_i, confhat_i = y_true_i[4], y_pred_i[9]

                        # equation 1
                        self.bbox_xy_offset_loss += self.lambda_coord * self.compute_xy_offset_loss(x_i, xhat_i, y_i, yhat_i)

                        # make them abs as sometimes the preds can be negative if no sigmoid layer.
                        # add 1e-6 for stability
                        self.bbox_wh_loss += self.lambda_coord * self.compute_wh_loss(w_i, what_i, h_i, hhat_i)

                        self.object_conf_loss += self.compute_object_conf_loss(conf_i, confhat_i)

                        # mention 2 other ways
                        # we can set conf_i = iou_b1 as well as the smaller of the two should be optimized to say there exist no object instead of proposing something.
                        # we can set conf_i = 0 as well and it will work.
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=torch.tensor(0., device=self.device), confhat_i=y_pred_i[4])

                    self.class_loss += self.compute_class_loss(p_i, phat_i)
                else:
                    # equation 4
                    # recall if there is no object, then conf_i is 0 by definition i.e. y_true_i[4] = 0
                    # confhat_i is y_pred_i[4] and y_pred_i[9] 
                    # it is worth noting that we loop over both bhat_1 and bhat_2 and calculate
                    # the no object loss for both of them. This is because we want to penalize
                    # the model for predicting an object when there is no object **for both bhat_1 and bhat_2**.
                    # in paper it seems that they only penalize for bhat_i^{\jmax} since they used 1_{i}^{noobj} notation
                    for j in range(self.B):
                        self.no_object_conf_loss += self.lambda_noobj * self.compute_no_object_conf_loss(conf_i=y_true_i[4], confhat_i=y_pred_i[4 + j * 5])

            print("\n")
            break
        total_loss = (
            self.bbox_xy_offset_loss
            + self.bbox_wh_loss
            + self.object_conf_loss
            + self.no_object_conf_loss
            + self.class_loss
        )

        # if dataloader has a batch size of 2, then our total loss is the average of the two losses.
        # i.e. total_loss = (loss1 + loss2) / 2 where loss1 is the loss for the first image in the
        # batch and loss2 is the loss for the second image in the batch.
        total_loss_averaged_over_batch = total_loss / batch_size
        # print(f"total_loss_averaged_over_batch {total_loss_averaged_over_batch}")

        return total_loss_averaged_over_batch


if __name__ == "__main__":
    torch.set_printoptions(threshold=5000)
    np.set_printoptions(threshold=5000)
    batch_size = 4
    S, B, C = 7, 2, 20
    lambda_coord, lambda_noobj = 5, 0.5

    # load directly the first batch of the train loader
    y_trues = torch.load("y_trues.pt")
    y_preds = torch.load("y_preds.pt")

    # print(y_trues.shape)
    # print(y_trues[0][3,3,:])

    criterion = YOLOv1Loss2D(S, B, C, lambda_coord, lambda_noobj)
    loss = criterion(y_trues, y_preds)
