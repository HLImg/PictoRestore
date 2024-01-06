# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : initializer
# @Email : lianghao@whu.edu.cn

import torch
import random
import diffusers
import numpy as np
import torch.optim as optim


def set_seed(seed=2017):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(name, net, params):
    if not isinstance(params, dict):
        raise TypeError('params must be a dict')
    params['params'] = net.parameters()
    if name.lower() == 'adam':
        optimizer = optim.Adam(**params)
    elif name.lower() == 'sgd':
        optimizer = optim.SGD(**params)
    elif name.lower() == 'adamw':
        optimizer = optim.AdamW(**params)
    else:
        raise ValueError(f"Only support Adam, SGD and AdamW, but received {name}")
    return optimizer


def get_scheduler(optimizer, params):
    if not isinstance(params, dict):
        raise TypeError('params must be a dict')
    params['optimizer'] = optimizer
    params['name'] = params['name'].lower()
    return diffusers.optimization.get_scheduler(**params)


def get_scheduler_torch(optimizer, params):
    name = params.pop('name').lower()
    params['optimizer'] = optimizer
    if name.lower() == 'CosineAnnealingLR'.lower():
        scheduler = optim.lr_scheduler.CosineAnnealingLR(**params)
    elif name.lower() == 'CosineAnnealingWarmRestarts'.lower():
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(**params)
    elif name.lower() == 'MultiStepLR'.lower():
        scheduler = optim.lr_scheduler.MultiStepLR(**params)
    else:
        raise ValueError(f"scheduler named {name} is not exits")
    return scheduler