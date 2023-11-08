# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:20
# @File    :   image_loss.py.py
# @Email   :   lianghao@whu.edu.cn

import torch
import torch.nn as nn

class SAMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SAMLoss, self).__init__()
        self.weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        if target.ndimension() == 5:
            target = target[:, 0, ...]
        if pred.ndimension() == 5:
            pred = pred[:, 0, ...]

        sum1 = torch.sum(target * pred, 1)
        sum2 = torch.sum(target * target, 1)
        sum3 = torch.sum(pred * pred, 1)

        t = (sum2 * sum3) ** 0.5
        num_local = torch.gt(t, 0)
        num = torch.sum(num_local)

        t = sum1 / t
        angle = torch.acos(t)
        sum_angle = torch.where(torch.isnan(angle),
                                torch.full_like(angle, 0),
                                angle).sum()

        if num == 0:
            avg_angle = sum_angle
        else:
            avg_angle = sum_angle / num

        sam = avg_angle * 180 / 3.14159256
        return self.weight * sam