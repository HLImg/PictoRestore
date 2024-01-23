# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:16
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .denoising import *

from src.utils import ARCH_REGISTRY

def build_arch(config):
    cfg = config['arch'].copy()
    name = cfg.pop('name')
    
    return ARCH_REGISTRY.get_obj(name)(**cfg)