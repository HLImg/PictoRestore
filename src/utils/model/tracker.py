# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:54
# @Author : Liang Hao
# @FileName : tracker.py
# @Email : lianghao@whu.edu.cn

import os
import shutil

import yaml
import socket
import logging

from datetime import datetime
from accelerate.tracking import (on_main_process, GeneralTracker, listify)


class HostnameFilter(logging.Filter):
    hostname = socket.gethostname()

    def filter(self, record):
        record.hostname = self.hostname
        return True


class Tracker(GeneralTracker):
    name = "tensorboard"
    requires_logging_directory = True
    main_process_only = True

    @on_main_process
    def __init__(self, config, verbose=False, **kwargs):
        try:
            from torch.utils import tensorboard
        except ModuleNotFoundError:
            import tensorboardX as tensorboard

        super().__init__()

        root_dir = config['root_dir']
        run_name = config['run_name']
        
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.exp_dir = os.path.join(root_dir, run_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        
        self.ckpt_dir = os.path.join(self.exp_dir, "save_state")
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        
        self.best_ckpt = os.path.join(self.exp_dir, "best_ckpt")
        if not os.path.exists(self.best_ckpt):
            os.mkdir(self.best_ckpt)

        config['exp_dir'] = self.exp_dir
        config['save_state'] = self.ckpt_dir
        config['best_ckpt'] = self.best_ckpt

        self._logger = logging.getLogger(__name__)
        self._logger.propagate = verbose
        self._logger.setLevel(logging.INFO)

        log_name = os.path.join(self.exp_dir, f'{self.time_stamp}.log')
        fmt = '%(asctime)s %(hostname)s[%(process)d] %(levelname)s %(message)s'
        formatter = logging.Formatter(fmt)
        file_handler = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(HostnameFilter())
        self._logger.addHandler(file_handler)
        file_handler.setFormatter(formatter)
        file_handler.close()

        self.tracker_dir = os.path.join(self.exp_dir, 'tracker')
        if not os.path.exists(self.tracker_dir):
            os.mkdir(self.tracker_dir)
        self.tensorboard_dir = os.path.join(self.tracker_dir, str(self.time_stamp))
        self.writer = tensorboard.SummaryWriter(self.tensorboard_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
    
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()

        with open(os.path.join(self.tensorboard_dir, "hparams.yml"), 'w') as outfile:
            try:
                yaml.dump(values, outfile)
            except yaml.representer.RepresenterError:
                raise ValueError("yaml.representer.RepresenterError")

    @on_main_process
    def log(self, values, step, **kwargs):
        values = listify(values)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()

    @on_main_process
    def info(self, msg, **kwargs):
        self._logger.info(msg, **kwargs)

    @on_main_process
    def debug(self, msg, **kwargs):
        self._logger.debug(msg, **kwargs)

    @on_main_process
    def log_images(self, values, step, **kwargs):
        for k, v in values.items():
            self.writer.add_images(k, v, global_step=step, **kwargs)

    @on_main_process
    def finish(self):
        self.writer.close()

    @on_main_process
    def store_init_config(self, config):
        if config.get('hyparams', False):
            self.store_init_configuration(config["hyparams"])

        shutil.copy(config['file'], os.path.join(self.exp_dir, 'config.yml'))