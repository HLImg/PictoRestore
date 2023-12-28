# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : logging
# @Email : lianghao@whu.edu.cn

import os
import time
import shutil
import logging
import datetime

from accelerate.logging import get_logger
from accelerate.tracking import on_main_process


class Logger:
    def __init__(self, root_dir):
        self.logger = get_logger(__name__)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        log_name = os.path.join(root_dir, 'logger.log')
        log_file = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')
        log_file.setLevel(logging.DEBUG)
        log_terminal = logging.StreamHandler()
        log_terminal.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        )
        log_file.setFormatter(formatter)
        log_terminal.setFormatter(formatter)
        self.logger.addHandler(log_file)
        self.logger.addHandler(log_terminal)

    @on_main_process
    def info(self, msg):
        self.logger.info(msg)

