# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 14:09
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

from .image_metric import *

metric = {
    "image": {
        "psnr": PSNR,
        "ssim": SSIM,
        "sam": SAM
    },
}

class Metric:
    def __init__(self, config):
        info = config['val']['metric']
        name = info["name"].lower()
        self.metric = {}
        for key, param in info["param"].items():
            self.metric[key] = metric[name][key.lower()](**param)

    def __call__(self, *args, **kwargs):
        return self.metric
