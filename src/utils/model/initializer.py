# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:54
# @Author : Liang Hao
# @FileName : initializer.py
# @Email : lianghao@whu.edu.cn

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


def get_optimizer(net, config):
    name = config['model']['optim']['name'].lower()
    params = config['model']['optim']['param']

    if not isinstance(params, dict):
        raise TypeError('params must be a dict')

    params['params'] = net.parameters()

    if name == 'adam':
        optimizer = optim.Adam(**params)
    elif name == 'sgd':
        optimizer = optim.SGD(**params)
    elif name == 'adamw':
        optimizer = optim.AdamW(**params)
    else:
        raise ValueError(f"Only support Adam, SGD and AdamW, but received {name}")
    return optimizer


def get_scheduler(optimizer, config, num_gpu=1, num_nodes=1):
    if not config['model'].get('schedule', False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000000, gamma=1
        )
    else:
        params = config['model']['schedule'].copy()
        if not isinstance(params, dict):
            raise TypeError('params must be a dict')

        params['optimizer'] = optimizer
        params['name'] = params['name'].lower()

        if 'num_warmup_steps' in params.keys() or \
                'num_training_steps' in params.keys():
            params['num_warmup_steps'] = int(params['num_warmup_steps']) * num_nodes * num_gpu
            params['num_training_steps'] = int(params['num_training_steps']) * num_nodes * num_gpu
            scheduler = diffusers.optimization.get_scheduler(**params)
        else:
            scheduler = get_scheduler_torch(params)

    return scheduler


def get_scheduler_torch(params):
    name = params.pop('name')
    if name.lower() == 'CosineAnnealingLR'.lower():
        scheduler = optim.lr_scheduler.CosineAnnealingLR(**params)
    elif name.lower() == 'CosineAnnealingWarmRestarts'.lower():
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(**params)
    elif name.lower() == 'MultiStepLR'.lower():
        scheduler = optim.lr_scheduler.MultiStepLR(**params)
    elif name.lower() == 'StepLR'.lower():
        scheduler = optim.lr_scheduler.LinearLR(**params)
    else:
        raise ValueError(f"scheduler named {name} is not exits")
    return scheduler
