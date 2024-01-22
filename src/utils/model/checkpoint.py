# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:55
# @Author : Liang Hao
# @FileName : checkpoint.py
# @Email : lianghao@whu.edu.cn

import os
import torch
import shutil
import logging
import datetime
import coloredlogs
import numpy as np

from accelerate.state import PartialState

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

state = PartialState()

@state.on_local_main_process
def backup(root_dir, source_input):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(root_dir, f"resume_{time_stamp}")
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.isdir(source_input):
        shutil.copytree(source_input, path)
    elif os.path.isfile(source_input, path):
        shutil.copy(source_input, path)
    else:
        raise ValueError(f"source input must be a file or directory, but received {source_input}")

    logger.info(f"Backup complete, source {source_input} >> target {path}")


def load_state_accelerate(accelerator, root_dir, resume_state):
    """
    Resuming training from the saved state.
    :param accelerator: the object of Accelerator
    :param root_dir: the root directory where the experimental results and outputs are saved.
    :param resume_state:  The directory from which to resume the training state,
                    previously saved using `accelerate.save_state()`.
    :return: {
                'last_epoch': ~,
                'last_iter': ~,
                'step_count': ~,
                'last_lr': ~
             }
    """
    logger.warning("[load_state_accelerate] indicates that training should be"
                   "fully resumes by Accelerate, ensuring that evey component (model, "
                   "optimizer, lr_scheduler, etc) remains unchanged.")
    if not os.path.isdir(resume_state):
        raise ValueError("resume_state is not a directory")

    accelerator.load_state(resume_state)
    
    backup(root_dir, source_input=resume_state)

    custom_check = os.path.join(resume_state, 'custom_checkpoint_0.pkl')
    if not os.path.exists(custom_check):
        raise ValueError(f"the custom checkpoint [{custom_check}] for Accelerate is not found")
    custom_register = torch.load(custom_check)

    ret = dict(
        last_epoch=custom_register['last_epoch'],
        last_iter=custom_register['last_epoch'],
        step_count=custom_register['_step_count'],
        last_lr=custom_register['_last_lr']
    )

    return ret



@state.on_local_main_process
def save_state_accelerate(accelerator, save_dir, save_name):
    save_path = os.path.join(save_dir, save_name)
    accelerator.save_state(save_path, safe_serialization=False)
    logger.info(f"Saves all training-related states using Accelerate to the specified path: {save_path}.")
