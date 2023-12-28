# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : __init__
# @Email : lianghao@whu.edu.cn

from .common import *
from .transforms import *
from .image_denoising import *
from .super_resolution import *

from src.utils import DATASET_REGISTRY

def get_dataset(config):
    ret = {}
    data_cfg = config['data']
    for mode in data_cfg.keys():
        ret[mode] = DATASET_REGISTRY.get_obj(
            obj_name=data_cfg[mode]['name']
        )(**data_cfg[mode]['param'])

    return ret

