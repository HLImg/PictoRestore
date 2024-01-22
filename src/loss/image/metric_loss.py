# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:11
# @Author : Liang Hao
# @FileName : metric_loss.py
# @Email : lianghao@whu.edu.cn

import torch
import numpy as np
import torch.nn as nn

from src.utils import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean', toY=False):
        super().__init__()
        assert reduction == 'mean'
        self.weight = weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
    
    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.

        assert len(pred.size()) == 4

        return self.weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()