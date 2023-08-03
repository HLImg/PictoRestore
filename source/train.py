# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:45
# @File    :   train.py
# @Email   :   lianghao@whu.edu.cn

import os
import torch
import shutil

from source.model import Model
from source.utils.common.log_util import Recorder, Logger

def train(model, config, logger, print_log):
    val_sum = len(model.test_loader)
    cur_iter = 0
    best_psnr = {
        "iter": 0,
        "psnr": 0
    }
    for epoch in range(model.start_epoch, model.end_epoch):
        model.net_g.train()
        for _, data in enumerate(model.train_loader):
            cur_iter += 1
            model.__feed__(data)

            if cur_iter % model.val_freq == 0 and print_log:
                # 不进行分布式验证，在一个gpu上完成验证
                model.net_g.eval()
                res_metric = {}
                with torch.no_grad():
                    for _, val_data in enumerate(model.test_loader):
                        res = model.__eval__(data)
                        # update
                        for key in res.keys():
                            res_metric[key] = res_metric.get(key, 0) + res[key]

                    for key in res_metric:
                        res_metric[key] = res_metric[key] / val_sum

                    if res_metric["psnr"] > best_psnr["psnr"]:
                        best_psnr["psnr"] = res_metric["psnr"]
                        best_psnr["iter"] = cur_iter

                    # 打印信息
                    info = f"epoch : {epoch}, cur_lr : {model.optimizer.param_groups[0]['lr']} \n"
                    for key in res_metric.keys():
                        info += f"cur_{key} : {res_metric[key] : .6f}"
                    info += '\n' + f"Best-PSNR {best_psnr['psnr'] : .6f} @ iter {best_psnr['iter']}"

                    logger.info(info)

            if cur_iter % model.save_freq == 0 and print_log:
                # TODO : 保存模型参数
                pass

    if print_log:
        logger.info("end training")

def main(config, args, accelerator):
    mm = True if config['train']['num_node'] > 1 else False
    print_flag = (not mm and accelerator.is_local_main_process) or (mm and accelerator.is_main_process)

    if print_flag:
        recoder = Recorder(config)
        _, yaml_file = os.path.split(args.yaml)
        shutil.copy(args.yaml, os.path.join(recoder.main_dir, yaml_file))
        logger = Logger(recoder.main_dir)()
        logger.info("start training")

    accelerator.wait_for_everyone()

    model = Model(config, accelerator)

    train(model, config, logger, print_flag)





