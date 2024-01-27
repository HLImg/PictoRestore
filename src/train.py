# -*- coding: utf-8 -*-
# @Time : 2024/01/22 19:53
# @Author : Liang Hao
# @FileName : train.py
# @Email : lianghao@whu.edu.cn

import os
import torch

from tqdm import tqdm
from .models import build_model
from .utils import Tracker
from accelerate.state import PartialState

state = PartialState()


def train(model, tracker):
    tracker.info(f"[{model.device}]: Start training, please note that it may take some time")
    model.accelerator.wait_for_everyone()

    for epoch in range(model.start_epoch, model.end_epoch):
        loop_train = tqdm(model.loader_train,
                          desc='training',
                          disable=not state.is_main_process)

        model.net.train()
        model.__reset__()
        
        for idx, data in enumerate(loop_train, 0):
            loop_train.set_description(f"[Epoch: {epoch} / {model.end_epoch}]")

            model.__feed__(idx=idx,
                           data=data,
                           pbar=loop_train,
                           tracker=tracker)

            if model.num_test > 0 and model.cur_iter % model.test_freq == 0:
                model.net.eval()
                model.__eval__(tracker=tracker)
                model.net.train()
                if model.main_process_only:
                    model.accelerator.wait_for_everyone()

            if model.cur_iter % model.save_freq == 0:
                model.save_state(save_name=f"save_state_{model.cur_iter}")

    tracker.info(f"{model.device}: Finish training")
    tracker.finish()


def main(args, config, accelerator):
    with accelerator.local_main_process_first():
        tracker = Tracker(config=config, verbose=args.verbose)
        tracker.store_init_config(config=config)
    accelerator.wait_for_everyone()
    
    if accelerator.num_processes == 1:
        args.eval_ddp = False
    
    trainer = build_model(accelerator=accelerator, config=config, main_process_only=not args.eval_ddp)
    
    train(model=trainer, tracker=tracker)