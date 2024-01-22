# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:10
# @Author : Liang Hao
# @FileName : feature.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.image as metric

from src.utils import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TotalVariation(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        """
        TV(Total Variation) Loss is often used as a regular term
        in the overall loss function to constrain the neural network,
        which can effectively propmote the spatial smoothness of the
        output results.

        :param beta:
        """
        super().__init__()
        self.weight = weight
        self.tv = metric.TotalVariation(reduction=reduction)

    def forward(self, inputs):
        return self.weight * self.tv(inputs)

