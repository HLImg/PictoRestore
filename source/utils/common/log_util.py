# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 14:27
# @File    :   log_util.py
# @Email   :   lianghao@whu.edu.cn

import os
import time
import shutil
import logging
import datetime

class Logger:
    def __init__(self, log_dir):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        while not os.path.exists(log_dir):
            time.sleep(0.1)
        log_name = os.path.join(log_dir, 'logger.log')
        log_file = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')
        log_file.setFormatter(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s %(asctime)s >> %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)

    def __call__(self):
        return self.logger

class Recorder:
    def __init__(self, config, main_flag):
        self.config = config
        self.main_flag = main_flag
        self.action = config["global"]["action"]
        self.info = config["global"]["dir_record"]
        self.main_dir = self.info["main_dir"]
        self._check_dir(self.main_dir)
        self.main_dir = os.path.join(self.main_dir, config["global"]["task"])
        self._check_dir(self.main_dir)
        self.main_dir = os.path.join(self.main_dir, self.info["main_name"])
        self._check_dir(self.main_dir)
        self.main_dir = os.path.join(self.main_dir, self._current_time())
        self._check_dir(self.main_dir)
        self.sub_dirs = {}
        for key, name in self.info["sub_dir"].items():
            self.sub_dirs[key] = os.path.join(self.main_dir, name)
            self._check_dir(self.sub_dirs[key])

        # self.__copy_file__()


    def __copy_file__(self):
        if self.config[self.action]['resume']['state'] and self.main_flag:
            state_file = self.config[self.action]['resume']['ckpt']
            _, name = os.path.split(state_file)
            save_dir = self.sub_dirs["resume_ckpt"]
            save_path = os.path.join(save_dir, name)
            shutil.copyfile(state_file, save_path)

    def _check_dir(self, *dirs):
        for dir in dirs:
            if not os.path.exists(dir) and self.main_flag:
                os.mkdir(dir)

    def _current_time(self):
        dtime = datetime.datetime.now()
        time_str = str(dtime.month).zfill(2) + str(dtime.day).zfill(2) + '_' + str(dtime.hour).zfill(2) + '_' + str(
            dtime.minute).zfill(2)
        return time_str

