# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:41
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

from .denoise import *

nets = {
    "denoise": {
        "nafnet": Nafnet,
    }
}

class Network:
    def __init__(self, config):
        info = config["net"]
        self.task = info["task"].lower()
        self.net_g_name = info["net_g"]["name"].lower()
        self.net_g_param = info["net_g"]["param"]

    def __call__(self, *args, **kwargs):
        net_g = nets[self.task][self.net_g_name](**self.net_g_param)
        return net_g