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

            tracker.log(values={'train/loss': model.loss.item()},
                        step=cur_iter)

            if (model.test_num > 0) and (cur_iter % model.val_freq == 0 or cur_iter == 1000):
                model.cur_metric = {}
                for _, val_data in enumerate(model.test_dataloader):
                    model.__eval__(val_data)

                for key, val in model.cur_metric.items():
                    model.cur_metric[key] /= model.test_num

                is_update, info_update = model.__update_metric__(cur_iter)
                tracker.log(model.cur_iter, step=cur_iter)
                tracker.info(msg=info_update)

                if is_update:
                    model.save_state(save_name=f"best_{cur_iter}")

            if cur_iter % model.save_freq == 0:
                model.save_state(save_name=f"save_state_{cur_iter}")
                tracker.info(f"{model.device}: save training state, {cur_iter}")

        tracker.info(f"{model.device}: Finish training")
        tracker.finish()


def main(args, config, accelerator):
    trainer = get_model(accelerator=accelerator, config=config)
    tracker = Tracker(config=config, verbose=args.verbose)
    tracker.store_init_config(config=config)
    train(model=trainer, tracker=tracker)