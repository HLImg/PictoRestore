# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:10
# @Author : Liang Hao
# @FileName : pixel.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class L1(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        return self.weight * F.l1_loss(input=inputs,
                                       target=targets,
                                       reduction=self.reduction, **kwargs)


@LOSS_REGISTRY.register()
class MSE(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        return self.weight * F.mse_loss(input=inputs,
                                        target=targets,
                                        reduction=self.reduction, **kwargs)

@LOSS_REGISTRY.register()
class Charbonnier(nn.Module):
    def __init__(self, epsilon=1e-3, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs, targets, **kwargs):
        loss = torch.sqrt((inputs - targets) ** 2 + self.epsilon)
        return self.weight * torch.mean(loss)


@LOSS_REGISTRY.register()
class Huber(nn.Module):
    def __init__(self, delta=1.0, weight=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        return self.weight * F.huber_loss(input=inputs,
                                          target=targets,
                                          reduction=self.reduction,
                                          delta=self.delta)


@LOSS_REGISTRY.register()
class SmoothL1(nn.Module):
    def __init__(self, beta=1.0, weight=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        return self.weight * F.smooth_l1_loss(input=inputs,
                                              target=targets,
                                              reduction=self.reduction,
                                              beta=self.beta, **kwargs)