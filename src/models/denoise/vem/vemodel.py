# -*- coding: utf-8 -*-
# @Time : 2024/01/23 14:24
# @Author : Liang Hao
# @FileName : vemodel.py
# @Email : lianghao@whu.edu.cn

import math
import torch
import logging
import coloredlogs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from src.utils import MODEL_REGISTRY
from src.datasets.transforms import ToImage
from src.models.abstract import StandardModel
from accelerate.tracking import on_main_process
from src.utils import get_optimizer, get_scheduler, load_state_accelerate

logger = logging.getLogger()
coloredlogs.install(level=logging.INFO)

@MODEL_REGISTRY.register()
class VEMModel(StandardModel):
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        
        self.cur_iter = 0
        
        # gpu information
        self.device = accelerator.device
        self.num_gpu = accelerator.num_processes
        
        self.__parser__(config)
        self.__build__(config)
        
        self.optimizer_theta = get_optimizer(self.net.net_theta, config=config)
        self.optimizer_phi = get_optimizer(self.net.net_phi, config=config)
        
        self.scheduler_theta = get_scheduler(
            self.optimizer_theta, config=config, num_gpu=self.num_gpu, num_nodes=self.num_nodes
        )
        
        self.scheduler_phi = get_scheduler(
            self.optimizer_phi, config=config, num_gpu=self.num_gpu, num_nodes=self.num_nodes
        )
        
        # prepare
        
        self.criterion_phi = nn.MSELoss()
        self.criterion_theta = nn.MSELoss()
        
        self.net, self.loader_train, self.criterion_phi, self.criterion_theta = self.accelerator.prepare(
            self.net, self.loader_train, self.criterion_phi, self.criterion_theta
        )
        
        self.optimizer_phi, self.optimizer_theta = self.accelerator.prepare(
            self.optimizer_phi, self.optimizer_theta
        )
        
        self.scheduler_phi, self.scheduler_theta = self.accelerator.prepare(
            self.scheduler_phi, self.scheduler_theta
        )
        
        if not self.main_process_only:
            logger.warning(f"[{self.device}]: Distributed Evaluation: All test data must have identical shapes.")
            self.loader_test = self.accelerator.prepare(self.loader_test)
        else:
            logger.warning(f"[{self.device}]: Evaluate Exclusively on the (Local) Main Process")
        
        self.accelerator.register_for_checkpointing(self.scheduler_phi)
        logger.info(f"[{self.device}]: Successfuly prepare relative tools for training.")
    
        self.auto_resume(config)
        
        iters_per_epoch = math.ceil(self.num_train / (self.batch_size * self.num_gpu * self.num_nodes))
        self.cur_iter = self.resume_info['last_epoch'] + 1 if self.resume_info is not None else 0
        self.start_epoch = math.ceil(self.cur_iter / iters_per_epoch)
        self.end_epoch = math.ceil(self.total_iters / iters_per_epoch)
        
        self.tensor2image = ToImage(out_type=np.uint8, min_max=(0, 1), rgb2bgr=True)
    
    
    def __feed__(self, idx, data, pbar, tracker):
        if (self.cur_iter + 1) % 2 == True:
            # E-Step
            self.optimizer_phi.zero_grad()
            lq, hq, sigma = data['lq'], data['hq'], data['sigma']
            est_x, _ = self.net(lq, step='E')
            loss_den = self.criterion_phi(est_x, hq)
            self.accelerator.backward(loss_den)
            self.optimizer_phi.step()
            self.scheduler_phi.step()
            
            if (self.cur_iter + 1) % 50 == 0:
                tracker.log(values={'train/loss_den': loss_den.item()}, step=self.cur_iter + 1)
            
        else:
            # M-Step
            self.optimizer_theta.zero_grad()
            lq, hq, sigma = data['lq'], data['hq'], data['sigma']
            est_x, est_sigma = self.net(lq, step='M')
            loss_est = self.criterion_theta(est_sigma, sigma)
            self.accelerator.backward(loss_est)
            self.optimizer_theta.step()
            self.scheduler_theta.step()
            
            if (self.cur_iter + 1) % 31 == 0:
                tracker.log(values={'train/loss_est': loss_est.item()}, step=self.cur_iter + 1)
        
        self.cur_iter = self.cur_iter + 1
    
    @on_main_process
    @torch.no_grad()
    def __eval__(self, tracker):
        metric = defaultdict(float)
        est_loss = 0.
        for _, data in enumerate(self.loader_test):
            lq, hq, sigma = data['lq'], data['hq'], data['sigma']
            denoise, est_sigma = self.net(lq, mode='')
            
            if not self.main_process_only:
                denoise, hq = self.accelerator.gather_for_metrics((denoise, hq))
            
            est_loss += F.mse_loss(est_sigma, sigma, reduction='sum')
            
            for i in range(denoise.shape[0]):
                tmp = self.metric(
                    self.tensor2image(denoise[i, ]),
                    self.tensor2image(hq[i, ])
                )
                
                for key, value in tmp.items():
                    metric[key] = metric[key] + value
        
        tracker.log(values={'val/loss_est': est_loss.mean()}, step=self.cur_iter)
        
        for key in metric.keys():
            metric[key] = metric[key] / self.num_test
        
        self.update_metric(metric, tracker=tracker, prefix='val')