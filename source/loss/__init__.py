# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:18
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

from .pixel_loss import *
from .image_loss import *

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
    },
    'image': {
        'sam': SAMLoss
    }
}

class Loss:
    def __init__(self, config):
        info = config["loss"]
        self.pixel_loss = None
        if 'pixel' in info.keys():
            self.pixel_loss = losses['pixel'][info['pixel']['name']](**info['pixel']['param'])

        self.image_loss = None
        if 'image' in info.keys():
            self.image_loss = losses['image'][info['image']['name']](**info['image']['param'])

    def __call__(self):
        if self.pixel_loss and not self.image_loss:
            return self.pixel_loss
        elif not self.pixel_loss and self.image_loss:
            return self.image_loss
        elif self.pixel_loss and self.image_loss:
            return lambda pred, target: self.pixel_loss(pred, target) + self.image_loss(pred, target)
        else:
            raise ValueError("the loss is not exits")