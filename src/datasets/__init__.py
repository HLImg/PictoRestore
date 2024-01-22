# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:22
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .common import *
from .transforms import *

from .image_denoising import *

from collections import OrderedDict
from src.utils import DATASET_REGISTRY

def build_dataset(config):
    ret = OrderedDict()
    cfg = config['data']
    
    for mode, params in cfg.items():
        name = params['name']
        param = params['param']
        ret[mode] = DATASET_REGISTRY.get_obj(name)(**param)
    
    return ret