# -*- coding: utf-8 -*-
# @Time : 2023/12/31
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .image_recon import *
from src.utils import METRIC_REGISTRY


def get_metric(config):
    ret = {}
    metric_cfg = config['metric']
    for name in metric_cfg.keys():
        ret[name] = METRIC_REGISTRY.get_obj(name)(**metric_cfg[name])

    def calculate_metric(input, target) -> dict:
        res = {}
        for name, metric in ret.items():
            res[name] = metric(input, target)
        return res

    return calculate_metric
