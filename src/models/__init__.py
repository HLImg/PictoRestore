# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:32
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

from .abstract import *

from src.utils import MODEL_REGISTRY

def build_model(accelerator, config):
    name = config['model']['name']
    params = dict(
        accelerator = accelerator,
        config = config
    )
    
    return MODEL_REGISTRY.get_obj(name)(**params)