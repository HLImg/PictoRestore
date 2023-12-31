# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : __init__
# @Email : lianghao@whu.edu.cn

from .denoising import *

from src.utils import ARCH_REGISTRY

def get_arch(config):
    # TODO: multiple net_arch for complex vision tasks
    ret = {}
    arch_cfg = config['arch']

    for name in arch_cfg.keys():
        ret[name] = ARCH_REGISTRY.get_obj(
            obj_name=arch_cfg[name]['name']
        )(**arch_cfg[name]['param'])

    return  ret
