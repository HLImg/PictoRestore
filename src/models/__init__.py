# -*- coding: utf-8 -*-
# @Time : 2024/1/1
# @Author : Liang Hao
# @FileName : __init__
# @Email : lianghao@whu.edu.cn

from .common import *

from src.utils import MODEL_REGISTRY

def get_model(accelerator, config):
    name = config['model']['name']
    params = dict(
        accelerator=accelerator, config=config
    )
    return MODEL_REGISTRY.get_obj(name)(**params)
