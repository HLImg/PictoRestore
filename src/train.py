# -*- coding: utf-8 -*-
# @Time : 2024/1/1
# @Author : Liang Hao
# @FileName : train
# @Email : lianghao@whu.edu.cn

import os
import torch

from tqdm import tqdm
from .models import get_model
from .utils import Tracker, backup
from accelerate.state import PartialState

state = PartialState()


def train(model, tracker):
    tracker.info(f"[{model.device}]: Start training, please note that it may take some time")
    cur_iter = model.cur_iter
    model.accelerator.wait_for_everyone()
    for epoch in range(model.start_epoch, model.end_epoch):
        loop_train = tqdm(model.train_dataloader,
                          desc="training",
                          disable=not model.accelerator.is_main_process)
        model.net_g.train()
        for _, data in enumerate(loop_train, 0):
            model.__feed__(data)
            cur_iter += 1

            loop_train.set_description(f"[Epoch: {epoch} / {model.end_epoch}]")
            loop_train.set_postfix(lr=model.get_cur_lr())

            tracker.log(values={'train/loss': model.loss},
                        step=cur_iter)

            if (model.test_num > 0) and (cur_iter % model.val_freq == 0 or cur_iter == 1000):
                if model.is_eval_ddp:
                    model.__eval_ddp__(cur_iter, tracker)
                else:
                    model.__eval_local__(cur_iter, tracker)
                
                model.__update_metric__(cur_iter, tracker)

            if cur_iter % model.save_freq == 0:
                model.save_state(save_name=f"save_state_{cur_iter}")
                tracker.info(f"{model.device}: save training state, {cur_iter}")

    tracker.info(f"{model.device}: Finish training")
    tracker.finish()


def main(args, config, accelerator):
    with accelerator.local_main_process_first():
        tracker = Tracker(config=config, verbose=args.verbose)
        tracker.store_init_config(config=config)
        
    accelerator.wait_for_everyone()
    
    trainer = get_model(accelerator=accelerator, config=config)
    
    train(model=trainer, tracker=tracker)