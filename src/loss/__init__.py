# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:01
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .image import *

from collections import OrderedDict
from src.utils import LOSS_REGISTRY

def build_loss(config):
    ret = OrderedDict()
    cfg = config['loss']
    
    for name, param in cfg.items():
        ret[name] = LOSS_REGISTRY.get_obj(name)(**param)
    
    def calculate(inputs, targets):
        res = 0.
        for _, loss in ret.items():
            res += loss(inputs, targets)
        return res
    
    return calculate