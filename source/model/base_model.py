# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:01
# @File    :   base_model.py
# @Email   :   lianghao@whu.edu.cn

import os
import math
import torch
import torch.optim as optim
from source.loss import Loss
from source.net import Network
from source.metric import Metric
from source.dataset import DataSet
from torch.utils.data import DataLoader
from source.utils.image.transpose import tensor2img


class BaseModel:
    def __init__(self, config, accelerator):
        self.conf_train = config["train"]
        self.conf_val = config["val"]
        self.resume_info = config["train"]['resume']

        self.accelerator = accelerator

        self.num_nodes = self.conf_train['num_node']
        self.bacth_per_gpu = self.conf_train['batch_per_gpu']
        self.num_gpu_per_node = self.conf_train['num_gpu_per_node']

        self.val_freq = self.conf_val['val_freq']
        self.save_freq = self.conf_train['save_freq']
        self.print_freq = self.conf_train['print_freq']
        self.total_iter = self.conf_train['total_iters']
        
        self.loss = 0
        self.cur_iter = 0

        # 初始化dataset, network, criterion, optimiser, scheduler
        criterion = Loss(config)()
        dataset = DataSet(config)()
        self.val_sum = dataset['test'].__len__()
        train_loader = DataLoader(dataset['train'],
                                  batch_size=self.bacth_per_gpu,
                                  shuffle=True,
                                  num_workers=self.conf_train['num_worker'])
        test_loader = DataLoader(dataset['test'], batch_size=1, shuffle=False, num_workers=0)
        
        net_g = Network(config)()
        if self.resume_info['state'] and self.resume_info['mode'].lower() == 'net':
            iter_ = int(self.resume_info['model'].split('.')[0].split('_')[-1])
            self.cur_iter = iter_
            ckpt = torch.load(self.resume_info['model'], map_location=torch.device('cpu'))
            net_g.load_state_dict(ckpt['net'])
            
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
        
        
        
        if self.resume_info['state']:
            if self.resume_info['mode'].lower() == 'all':
                iter_ = int(self.resume_info['ckpt'].split('_')[-1])
                self.cur_iter = iter_
                self.accelerator.load_state(self.resume_info['ckpt'])
            elif self.resume_info['mode'].lower() == 'other':
                self.__resume_other__()
            
        
        # register the optimizer, schulder et al
        self.accelerator.register_for_checkpointing(self.optimizer)
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.accelerator.register_for_checkpointing(self.net_g)
        # ======================================================= #
        

        num_iter_per_epoch = math.ceil(len(dataset['train']) / (self.bacth_per_gpu * self.num_gpu_per_node * self.num_nodes))
        self.start_epoch = math.ceil(self.cur_iter / num_iter_per_epoch)
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
        self.optimizer.zero_grad()
        predicted = self.net_g(data['lq'])
        loss = self.criterion(predicted, data['hq'])
        # ========================================= #
        self.accelerator.backward(loss)
        # ========================================= #
        self.optimizer.step()
        self.scheduler.step()

    def __eval__(self, data):
        predicted = self.net_g(data['lq'])
        # ======================================== #
        all_predicts, all_targets = self.accelerator.gather_for_metrics((predicted, data['hq']))
        # ======================================== #
        res = {}
        for key, metric in self.metric.items():
            for ii in range(all_predicts.shape[0]):
                res[key] = res.get(key, 0.) + metric(tensor2img(all_predicts[ii]),
                                   tensor2img(all_targets[ii]))

        return res
    
    def __resume_other__(self):
        if self.resume_info.get('optim', False):
            self.accelerator.load_state(self.resume_info['optim'])
        if self.resume_info.get('model', False):
            self.accelerator.load_state(self.resume_info['model'])
                
    def save_train_states(self, path, cur_iter):
        save_dir = os.path.join(path, f'save_iter_{cur_iter}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.accelerator.save_state(save_dir)