# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:01
# @File    :   base_model.py
# @Email   :   lianghao@whu.edu.cn

import math
import torch.optim as optim
from source.loss import Loss
from source.net import Network
from source.metric import Metric
from source.dataset import DataSet
from torch.utils.data import DataLoader


class BaseModel:
    def __init__(self, config, accelerator):
        self.conf_train = config["train"]
        self.conf_val = config["val"]

        self.accelerator = accelerator

        self.num_nodes = self.conf_train['num_node']
        self.bacth_per_gpu = self.conf_train['batch_per_gpu']
        self.num_gpu_per_node = self.conf_train['num_gpu_per_node']

        self.val_freq = self.conf_val['val_freq']
        self.save_freq = self.conf_train['save_freq']
        self.total_iter = self.conf_train['total_iters']

        # 初始化dataset, network, criterion, optimiser, scheduler
        criterion = Loss(config)()
        dataset = DataSet(config)()
        train_loader = DataLoader(dataset['train'],
                                  batch_size=self.bacth_per_gpu,
                                  shuffle=True,
                                  num_workers=self.conf_train['num_worker'])
        test_loader = DataLoader(dataset['test'], batch_size=1, shuffle=False, num_workers=0)
        net_g = Network(config)()
        optimizer = self.step_optimizer(self.conf_train['optim']['optimizer']['name'],
                                        param=self.conf_train['optim']['optimizer']['param'],
                                        net_param=net_g.parameters())
        scheduler = self.step_scheduler(self.conf_train['optim']['scheduler']['name'],
                                       param=self.conf_train['optim']['scheduler']['param'],
                                       optimizer=optimizer)

        self.metric = Metric(config)()
        # ======================================================= #
        # accelerator进行加速配置
        self.train_loader = self.accelerator.prepare(train_loader)
        self.test_loader = self.accelerator.prepare(test_loader)
        self.net_g = self.accelerator.prepare(net_g)
        self.criterion = self.accelerator.prepare(criterion)
        self.optimizer = self.accelerator.prepare(optimizer)
        self.scheduler = self.accelerator.prepare(scheduler)
        # ======================================================= #

        num_iter_per_epoch = math.ceil(len(dataset['train']) / (self.bacth_per_gpu * self.num_gpu_per_node * self.num_nodes))
        self.start_epoch = 0
        self.end_epoch = math.ceil(self.total_iter / num_iter_per_epoch)

    def step_optimizer(self, name, param, net_param):
        param['params'] = net_param
        if name.lower() == 'adamw':
            optimizer = optim.AdamW(**param)
        elif name.lower() == 'adam':
            optimizer = optim.Adam(**param)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(**param)
        else:
            raise ValueError(f"optimizer named {name} is not exits")

        return optimizer

    def step_scheduler(self, name, param, optimizer):
        param['optimizer'] = optimizer
        if name.lower() == 'CosineAnnealingLR'.lower():
            scheduler = optim.lr_scheduler.CosineAnnealingLR(**param)
        elif name.lower() == 'CosineAnnealingWarmRestarts'.lower():
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(**param)
        elif name.lower() == 'MultiStepLR'.lower():
            scheduler = optim.lr_scheduler.MultiStepLR(**param)
        else:
            raise ValueError(f"scheduler named {name} is not exits")
        return scheduler

    def __feed__(self, data):
        pass

    def __eval__(self, data):
        pass

