# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:01
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .image_recon import *

from collections import OrderedDict
from src.utils import METRIC_REGISTRY


def build_metric(config):
    ret = OrderedDict()
    cfg = config['metric']

    for name, param in cfg.items():
        ret[name] = METRIC_REGISTRY.get_obj(name)(**param)
    
    def calculate(inputs, targets):
        res = OrderedDict()
        for name, metric in ret.items():
            res[name] = metric(inputs, targets)
        return res
    
    return calculate