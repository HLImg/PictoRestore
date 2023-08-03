# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:18
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

from .pixel_loss import *

pixel_loss = {
    "l1": L1Loss,
    "mse": MSELoss,
    "psnr": PSNRLoss
}

losses = {
    "pixel" : {
        "l1": L1Loss,
        "mse": MSELoss,
        "psnr": PSNRLoss
    }
}

class Loss:
    def __init__(self, config):
        info = config["loss"]
        self.pixel_loss = None
        if 'pixel' in info.keys():
            self.pixel_loss = losses['pixel'][info['pixel']['name']](**info['pixel']['param'])

    def __call__(self):
        if self.pixel_loss:
            return self.pixel_loss
        else:
            raise ValueError("the loss is not exits")