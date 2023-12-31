# -*- coding: utf-8 -*-
# @Time : 2023/12/31
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .classify import *
from .image import *

from src.utils import LOSS_REGISTRY


def get_loss(config):
    ret = {}
    loss_cfg = config['loss']
    for name in loss_cfg.keys():
        ret[name] = LOSS_REGISTRY.get_obj(name)(**loss_cfg[name])

    def cal_loss(inputs, targets):
        # TODO: if multiple loss functions are present, return individual loss
        #       components for plotting.
        res = 0.
        for _, loss in ret.items():
            res += loss(inputs, targets)
        return res

    return cal_loss