# -*- coding: utf-8 -*-
# @Time : 2024/01/24 11:59
# @Author : Liang Hao
# @FileName : pretrain.py
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
class VEModelPretrain(StandardModel):
    def __init__(self, config, accelerator, main_process_only):
        super().__init__(config, accelerator, main_process_only)
        
        logger.warning(f"main_process_only init: {self.main_process_only}")
        
    def __feed__(self, idx, data, pbar, tracker):
        with self.accelerator.accumulate(self.net):
            self.optimizer.zero_grad()
            lq, hq, sigma = data['lq'], data['hq'], data['sigma']
            
            denoise, est_sigma = self.net(lq)
            
            loss_denoise = self.criterion(denoise, hq)
            loss_estimate = self.criterion(est_sigma, sigma)
            loss = loss_denoise + loss_estimate
            
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            
            if (self.cur_iter + 1) % 50 == 0:
                tracker.log(values={
                                "train/loss_den": loss_denoise.item(),
                                "train/loss_est": loss_estimate.item()
                            }, 
                            step=self.cur_iter + 1)
        
        self.cur_iter += 1
        pbar.set_postfix(dst=loss_denoise.item(),
                         est=loss_estimate.item())
    
    @on_main_process
    @torch.no_grad()
    def __eval__(self, tracker):
        metric = defaultdict(float)
        for _, data in enumerate(self.loader_test):
            lq, hq, sigma = data['lq'], data['hq'], data['sigma']
            if self.main_process_only:
                lq, hq = lq.to(self.device), hq.to(self.device)
                sigma = sigma.to(self.device)
        
            denoise, est_sigma = self.net(lq)
            
            if not self.main_process_only:
                sigma, est_sigma = self.accelerator.gather_for_metrics((sigma, est_sigma))
                denoise, hq = self.accelerator.gather_for_metrics((denoise, hq))
            
            for i in range(denoise.shape[0]):
                metric['dst'] += F.mse_loss(denoise[i, ], hq[i, ]).item()
                metric['est'] += F.mse_loss(est_sigma[i, ], sigma[i, ]).item()
                
                tmp = self.metric(
                    self.tensor2image(denoise[i, ]),
                    self.tensor2image(hq[i, ])
                )
                
                for key, value in tmp.items():
                    metric[key] = metric[key] + value
            
        for key in metric.keys():
            metric[key] = metric[key] / self.num_test
        
        self.update_metric(metric, tracker=tracker, prefix='val')
            
            