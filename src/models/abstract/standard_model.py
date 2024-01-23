# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:54
# @Author : Liang Hao
# @FileName : standard_model.py
# @Email : lianghao@whu.edu.cn

import math
import torch
import logging
import coloredlogs
import numpy as np
import os.path as osp

from collections import defaultdict

from src.loss import build_loss
from src.arches import build_arch
from src.metrics import build_metric
from src.datasets import build_dataset
from accelerate.tracking import on_main_process

from src.utils import get_optimizer, get_scheduler, load_state_accelerate
from torch.utils.data import DataLoader

from .base_model import BaseModel
from src.utils import MODEL_REGISTRY

from src.datasets.transforms import ToImage

logger = logging.getLogger()
coloredlogs.install(level=logging.INFO)

@MODEL_REGISTRY.register()
class StandardModel(BaseModel):
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        
        """Necessary Setting"""
        self.__parser__(config)
        self.__build__(config)
        self.__prepare__()
        self.auto_resume(config)
        
        iters_per_epoch = math.ceil(self.num_train / (self.batch_size * self.num_gpu * self.num_nodes))
        self.cur_iter = self.resume_info['last_epoch'] + 1 if self.resume_info is not None else 0
        self.start_epoch = math.ceil(self.cur_iter / iters_per_epoch)
        self.end_epoch = math.ceil(self.total_iters / iters_per_epoch)
        
        """Other Setting"""
        
        self.tensor2image = ToImage(out_type=np.uint8, min_max=(0, 1), rgb2bgr=True)
    
    def __build__(self, config):
        self.net = build_arch(config)
        self.metric = build_metric(config)
        self.criterion = build_loss(config)
        
        self.optimizer = get_optimizer(self.net, config=config)
        self.scheduler = get_scheduler(
            self.optimizer, config=config, num_gpu=self.num_gpu, num_nodes=self.num_nodes
        )
        
        datasets = build_dataset(config=config)
        
        self.loader_train, self.loader_test = None, None
        
        loader = config['model']['loader']
        
        if 'train' in datasets.keys():
            self.num_train = len(datasets['train'])
            loader['train']['dataset'] = datasets['train']
            self.loader_train = DataLoader(**loader['train'])
            self.batch_size = loader['train']['batch_size']
        
        if 'test' in datasets:
            self.num_test = len(datasets['test'])
            loader['test']['dataset'] = datasets['test']
            self.loader_test = DataLoader(**loader['test'])
        
        logger.info(f"[{self.device}]: Successfuly build net, metric, criterion, optimizer, scheduler, and dataloader.")
        
    def __prepare__(self):
        self.net, self.optimizer, self.scheduler, self.loader_train, self.criterion = self.accelerator.prepare(
            self.net, self.optimizer, self.scheduler, self.loader_train, self.criterion
        )
        
        if not self.main_process_only:
            logger.warning(f"[{self.device}]: Distributed Evaluation: All test data must have identical shapes.")
            self.loader_test = self.accelerator.prepare(self.loader_test)
        else:
            logger.warning(f"[{self.device}]: Evaluate Exclusively on the (Local) Main Process")
        
        self.accelerator.register_for_checkpointing(self.scheduler)
        
        logger.info(f"[{self.device}]: Successfuly prepare relative tools for training.")
      
    def __parser__(self, config):
        self.exp_dir = osp.join(config['root_dir'], config['run_name'])
        self.ckpt_dir = osp.join(self.exp_dir, "save_state")
        self.best_ckpt = osp.join(self.exp_dir, "best_ckpt")
        
        self.num_nodes = int(config['model']['num_nodes'])
        self.save_freq = int(config['model']['save_freq'])
        self.train_freq = int(config['model']['train_freq'])
        self.test_freq = int(config['model']['test_freq'])
        self.total_iters = int(config['model']['iteration'])
        self.best_metric = config['model']['best_metric']
        
    
    def auto_resume(self, config):
        self.resume_info = None
        if not config['resume']:
            return
        
        self.resume_info = load_state_accelerate(
            accelerator=self.accelerator,
            root_dir=self.exp_dir,
            resume_state=config['resume_state']
        )
        
        logger.info(f"[{self.device}]: Successfully resume training states by accelerate")
        
    
    def __reset__(self):
        self.train_loss = 0
    
    def __feed__(self, idx, data, pbar, tracker):
        with self.accelerator.accumulate(self.net):
            self.optimizer.zero_grad()
            lq, hq = data['lq'], data['hq']
            predict = self.net(lq)
            loss = self.criterion(predict, hq)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
        
        self.cur_iter = self.cur_iter + 1
        pbar.set_postfix(loss=loss.item())
        
        if self.cur_iter % self.train_freq == 0:
            tracker.log(values={'train/loss': loss.item()}, step=self.cur_iter + 1)
        

    @on_main_process
    @torch.no_grad()
    def __eval__(self, tracker):
        metric = defaultdict(float)
        for _, data in enumerate(self.loader_test):
            lq, hq = data['lq'], data['hq']
            denoise = self.net(lq)
            
            if not self.main_process_only:
                denoise, hq = self.accelerator.gather_for_metrics((denoise, hq))
            
            for i in range(denoise.shape[0]):
                tmp = self.metric(
                    self.tensor2image(denoise[i, ]),
                    self.tensor2image(hq[i, ])
                )
                
                for key, value in tmp.items():
                    metric[key] = metric[key] + value
        
        for key in metric.keys():
            metric[key] = metric[key] / self.num_test
        
        self.update_metric(metric, tracker=tracker, prefix='val')
                
    
    @on_main_process
    def save_best(self, save_name):
        net_warp = self.accelerator.unwrap_model(self.net)
        path = osp.join(self.best_ckpt, f"{save_name}.pth")
        torch.save(net_warp.state_dict(), path)
    
    
    def update_metric(self, metric, tracker, prefix='val'):
        tf_values = {}
        for key, value in metric.items():
            if key == self.best_metric['name']:
                if self.best_metric['value'] < value:
                    self.best_metric['value'] = value
                    self.save_best(f"best_{self.cur_iter}.pth")
            
            tf_values[f"{prefix}/{key}"] = value
        
        tracker.log(values=tf_values, step=self.cur_iter)
    
    
    @on_main_process
    def save_state(self, save_name):
        save_path = osp.join(self.ckpt_dir, save_name)
        self.accelerator.save_state(save_path, safe_serialization=False)
        logger.info(f"[{self.device}]: Saves all training-related states using "
                    f"Accelerate to the specified path: {save_path}.")