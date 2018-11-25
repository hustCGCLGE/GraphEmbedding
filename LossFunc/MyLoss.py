import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self, w1, w2, w3, node_num):
        super(MyLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.node_num = node_num

    def forward(self, x_diff1, x_diff2, y_diff, a1, a2, x_ij):

        def weighted_mse_x(x_diff, a):
            b = a[0, 0:self.node_num]
            return (x_diff * b).sum(dim=-1) / a[0, self.node_num:]

        def weighted_mse_y(y_diff_p, x_ij_p):
            return y_diff_p.sum(dim=-1) * x_ij_p

        loss1 = weighted_mse_x(x_diff1, a1) * self.w1
        loss2 = weighted_mse_x(x_diff2, a2) * self.w2
        loss3 = weighted_mse_y(y_diff, x_ij) * self.w3
        return loss1 + loss2 + loss3


if __name__ == "__main__":

    x = MyLoss(0, 0, 0)
    MyLoss.forward(1, 1, 1, 1, 1, 1, 1)
