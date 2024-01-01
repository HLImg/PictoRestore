# -*- coding: utf-8 -*-
# @Time : 2023/12/31
# @Author : Liang Hao
# @FileName : base_model
# @Email : lianghao@whu.edu.cn

import os
import math
import torch
import logging
import coloredlogs

from torch.utils.data import DataLoader
from accelerate.tracking import on_main_process

from src.datasets import get_dataset, ToNdarray_chw2hwc
from src.archs import get_arch
from src.loss import get_loss
from src.metrics import get_metric
from src.utils.model import (get_optimizer, get_scheduler,
                             load_state_accelerate,
                             save_state_accelerate)
from src.utils import MODEL_REGISTRY

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


@MODEL_REGISTRY.register()
class BaseModel(object):
    def __init__(self, accelerator, config):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.num_gpu = accelerator.num_process

        self.loss = 0
        self.num_nodes = config['model']['num_nodes']
        self.save_freq = config['model']['save_freq']
        self.total_iters = config['model']['iteration']
        self.batch_size = config['model']['batch_size']

        # the directory for saving training results
        self.root_dir = os.path.join(config.exp_dir, config.run_name)
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        # the subdir for saving training states
        self.ckpt_dir = os.path.join(self.root_dir, "save_state")
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

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
        self.optimizer = get_optimizer(name=config['model']['optim']['name'],
                                       params=config['model']['optim']['param'],
                                       net=self.net_g)
        if not config['model'].get('schedule', False):

            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=[self.total_iters * self.num_gpu * 10 * self.num_nodes]
            )

        else:
            config['model']['schedule']['num_warmup_steps'] *= self.num_gpu * self.num_nodes
            config['model']['schedule']['num_training_steps'] *= self.num_gpu * self.num_nodes
            self.scheduler = get_scheduler(optimizer=self.optimizer,
                                           params=config['model']['schedule'])

        # loss function
        self.criterion = get_loss(config)

        # prepare for accelerate
        self.net_g, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.net_g, self.optimizer, self.scheduler
        )

        self.train_dataloader, self.test_dataloader, self.criterion = self.accelerator.prepare(
            self.train_dataloader, self.test_dataloader, self.criterion
        )

        self.accelerator.register_for_checkpointing(self.scheduler)

        logger.info("Successfully setup [Train/Test DataLoader, Net, Optimizer, Scheduler]")

        resume_info = None
        if config['model']['resume']['mode']:
            resume_info = load_state_accelerate(
                accelerator=self.accelerator,
                root_dir=self.root_dir,
                resume_state=config['model']['resume']['state']
            )
            logger.info("Successfully resume training states by accelerate")

        # metric
        self.metric = get_metric(config)

        iters_per_epoch = math.ceil(len(dataset['train'])) / (self.batch_size * self.num_gpu * self.num_nodes)
        self.cur_iter = resume_info['last_epoch'] + 1 if resume_info is not None else 0
        self.start_epoch = math.ceil(self.cur_iter / iters_per_epoch)
        self.end_epoch = math.ceil(self.total_iters / iters_per_epoch)
        self.trans2bhwc_np = ToNdarray_chw2hwc()

        self.best_metric = {}

    def __feed__(self, data):
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.net_g):
            lq, hq = data['lq'], data['hq']
            predicted = self.net_g(lq)
            loss = self.criterion(predicted, hq)
            self.loss = loss.item()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    @on_main_process
    def __eval__(self, data):
        lq, hq = data['lq'], data['hq']
        predicted = self.net_g(lq)
        all_predicted, all_target = self.accelerator.gather_for_metrics((predicted, hq))
        inputs = self.trans2bhwc_np(all_predicted)  # b h w c
        targets = self.trans2bhwc_np(all_target)  # b h w c
        res = {}
        for i in range(inputs.size(0)):
            tmp = self.metric(inputs[i,], targets[i,])
            for key, value in tmp.items():
                res[key] = res.get(key, 0) + value
        return res

    def save_state(self, save_name):
        save_state_accelerate(accelerator=self.accelerator,
                              save_dir=self.ckpt_dir,
                              save_name=save_name)

    def get_cur_lr(self):
        return self.optimizer.param_groups[0]['lr']
