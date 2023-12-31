# -*- coding: utf-8 -*-
# @Time : 2023/12/31
# @Author : Liang Hao
# @FileName : base_model
# @Email : lianghao@whu.edu.cn

import os
import math
import torch

from torch.utils.data import DataLoader

from src.datasets import get_dataset
from src.archs import get_arch
from src.loss import get_loss
from src.metrics import get_metric
from src.utils.model import setup_scheduler, setup_optimizer

class BaseModel(object):
    def __init__(self, accelerator, config):
        self.config = config
        self.device = accelerator.device
        self.num_gpu = accelerator.num_process

        self.save_freq = config['model']['save_freq']
        self.total_iters = config['model']['iteration']
        self.batch_size = config['model']['batch_size']

        # dataset
        dataset = get_dataset(config)
        self.train_dataloader = None
        self.test_dataloader = None
        if 'train' in dataset:
            self.train_dataloader = DataLoader(dataset['train'],
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=config['model']['num_worker'])

        if 'test' in dataset:
            self.test_dataloader = DataLoader(dataset['test'],
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

        # TODO multiple network
        self.net_g = get_arch(config)['net_g']

        # optimizer and scheduler
        self.optimizer = setup_optimizer(name=config['model']['optim']['name'],
                                         params=config['model']['optim']['param'],
                                         net=self.net_g)
        if not config['model'].get('schedule', False):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=[]
            )



    def __feed__(self, data):
        pass

    def __eval__(self, data):
        pass
