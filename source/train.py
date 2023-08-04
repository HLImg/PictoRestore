# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:45
# @File    :   train.py
# @Email   :   lianghao@whu.edu.cn

import os
import torch
import shutil
from tqdm import tqdm
from source.model import Model
from source.utils.common.log_util import Recorder, Logger


def train(model, recoder, logger, main_flag):
    val_sum = len(model.test_loader)
    cur_iter = 0
    best_psnr = {
        "iter": 0,
        "psnr": 0
    }
    for epoch in range(model.start_epoch, model.end_epoch):
        loop_train = tqdm(model.train_loader, desc='training', disable=not model.accelerator.is_local_main_process)
        model.net_g.train()
        for _, data in enumerate(loop_train, 0):
            cur_iter += 1
            model.__feed__(data)
            loop_train.set_description(
                f"epoch [{epoch} / {model.end_epoch}]")
            loop_train.set_postfix(lr=model.optimizer.param_groups[0]['lr'])

            if cur_iter % model.val_freq == 0 and main_flag:
                # 不进行分布式验证，在一个gpu上完成验证
                model.net_g.eval()
                res_metric = {}
                with torch.no_grad():
                    for _, val_data in enumerate(model.test_loader):
                        res = model.__eval__(data)
                        # update
                        for key in res.keys():
                            res_metric[key] = res_metric.get(key, 0) + res[key]

                    for key in res_metric.keys():
                        res_metric[key] = res_metric[key] / val_sum

                    if res_metric["psnr"] > best_psnr["psnr"]:
                        best_psnr["psnr"] = res_metric["psnr"]
                        best_psnr["iter"] = cur_iter

                    # 打印信息
                    info = f"epoch : {epoch}, cur_lr : {model.optimizer.param_groups[0]['lr'] : .6f} \n" + ' ' * 32
                    for key in res_metric.keys():
                        info += f"{key} : {res_metric[key] : .6f}    "
                    info += '\n' + ' ' * 32 + f"best psnr : {best_psnr['psnr'] : .6f} @ iter {best_psnr['iter']}"
                    logger.info(info)
                    # 保存
                    net_warp = model.accelerator.unwrap_model(model.net_g)
                    best_path = os.path.join(recoder.sub_dirs['best_ckpt'], f'best_iter_{cur_iter}.ckpt')
                    torch.save({'net': net_warp.state_dict()}, best_path)


            if cur_iter % model.save_freq == 0 and main_flag:
                # TODO : 保存模型参数
                net_warp = model.accelerator.unwrap_model(model.net_g)
                save_path = os.path.join(recoder.sub_dirs['save_ckpt'], f'save_iter_{cur_iter}.ckpt')
                torch.save({'net': net_warp.state_dict()}, save_path)


    if main_flag:
        logger.info("end training")


def main(config, args, accelerator):
    mm = True if config['train']['num_node'] > 1 else False
    main_flag = (not mm and accelerator.is_local_main_process) or (mm and accelerator.is_main_process)

    recoder = Recorder(config, main_flag)
    _, yaml_file = os.path.split(args.yaml)
    if main_flag:
        shutil.copy(args.yaml, os.path.join(recoder.main_dir, yaml_file))

    accelerator.wait_for_everyone()
    logger = Logger(recoder.main_dir)()

    if main_flag:
        logger.info("start training")

    model = Model(config, accelerator)()

    train(model, recoder, logger, main_flag)