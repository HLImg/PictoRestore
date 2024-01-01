# -*- coding: utf-8 -*-
# @Time : 2024/1/1
# @Author : Liang Hao
# @FileName : train
# @Email : lianghao@whu.edu.cn

import os
import torch

from tqdm import tqdm
from .utils import Tracker

def train(model, tracker):
    tracker.info(f"[{model.device}]: Start training, it will take a time")
    cur_iter = model.cur_iter
    for epoch in range(model.start_epoch, model.end_epoch):
        loop_train = tqdm(model.train_loader,
                          desc="training",
                          disable=not model.accelerator.is_main_process)
        model.net_g.train()
        for _, data in enumerate(loop_train, 0):
            model.__feed__(data)
            cur_iter += 1

            loop_train.set_description(f"[Epoch [{epoch} / {model.end_epoch}]]")
            loop_train.set_postfix(lr=model.get_cur_lr())

            tracker.log()
