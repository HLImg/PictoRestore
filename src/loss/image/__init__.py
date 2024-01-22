# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:10
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .pixel import L1, SmoothL1, MSE, Charbonnier, Huber
from .feature import TotalVariation
from .metric_loss import PSNRLoss