# -*- coding: utf-8 -*-
# @Time : 2023/12/29
# @Author : Liang Hao
# @FileName : checkpoint
# @Email : lianghao@whu.edu.cn

import os

import coloredlogs
import numpy as np
import logging

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

from accelerate.tracking import on_main_process

def load_state_accelerate(resume_state):
    logger.info("Must be a directory")
    if not os.path.isdir(resume_state):
        raise ValueError("resume_state is not a directory")

if __name__ == "__main__":
    load_state_accelerate("./src")
    print("Hello")
