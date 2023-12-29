# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : logging
# @Email : lianghao@whu.edu.cn
import colorlog
import os
import time
import yaml
import shutil
import logging
import coloredlogs

from datetime import datetime
from colorlog import ColoredFormatter
from accelerate.tracking import (on_main_process, GeneralTracker, listify)


class Tracker(GeneralTracker):
    name = "tensorboard"
    requires_logging_directory = True
    @on_main_process
    def __init__(self, run_name, save_dir, verbose=False, **kwargs):
        try:
            from torch.utils import tensorboard
        except ModuleNotFoundError:
            import tensorboardX as tensorboard

        super().__init__()

        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.root_dir = os.path.join(save_dir, run_name)

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        log_name = os.path.join(self.root_dir, f'{self.time_stamp}.log')


        stream_handler = colorlog.StreamHandler() if verbose else None
        file_handler = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')

        self._logger.addHandler(file_handler)
        if verbose:
            self._logger.addHandler(stream_handler)

        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        formatter_console = coloredlogs.ColoredFormatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            level_styles=dict(
                debug=dict(color='white'),
                info=dict(color='blue'),
                warning=dict(color='yellow', bright=True),
                error=dict(color='red', bold=True, bright=True),
                critical=dict(color='black', bold=True, background='red'),
            ),
            field_styles=dict(
                name=dict(color='yellow', bold=True),
                asctime=dict(color='green', bold=True),
                levelname=dict(color='red'),
                lineno=dict(color='white'),
            )
        )

        if verbose:
            stream_handler.setFormatter(formatter_console)
        file_handler.setFormatter(formatter)

        if verbose:
            stream_handler.close()
        file_handler.close()




        # log_name = os.path.join(self.root_dir, f'{self.time_stamp}.log')
        # log_file = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')
        # log_file.setLevel(logging.INFO)
        # # formatter = logging.Formatter(
        # #     '%(asctime)s - %(levelname)s: %(message)s'
        # # )
        # log_file.setFormatter(coloredFormatter)
        #
        # if verbose:
        #     log_terminal = logging.StreamHandler()
        #     log_terminal.setLevel(logging.INFO)
        #     log_terminal.setFormatter(coloredFormatter)
        #     self.logger.addHandler(log_terminal)
        # self.logger.addHandler(log_file)

        self.tracker_dir = os.path.join(self.root_dir, 'tracker')
        self.writer = tensorboard.SummaryWriter(self.tracker_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()

        dir_name = os.path.join(self.tracker_dir, str(self.time_stamp))
        os.makedirs(dir_name, exist_ok=True)

        with open(os.path.join(dir_name, "hparams.yml"), 'w') as outfile:
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
    def INFO(self, msg, **kwargs):
        self._logger.info(msg, **kwargs)

    @on_main_process
    def DEBUG(self, msg, **kwargs):
        self._logger.debug(msg, **kwargs)

    @on_main_process
    def log_images(self, values, step, **kwargs):
        for k, v in values.items():
            self.writer.add_images(k, v, global_step=step, **kwargs)

    @on_main_process
    def finish(self):
        self.writer.close()


