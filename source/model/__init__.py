# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:46
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn


from .denoise import *
from .common.standard import standardModel

models = {
    "denoise": {
        "standard": standardModel,
    }
}

class Model:
    def __init__(self, config, accelerator):
        info = config["model"]

        self.task = info["task"].lower()
        self.name = info["name"].lower()

        self.param = {
            'config': config,
            'accelerator': accelerator
        }

    def __call__(self, *args, **kwargs):
        if self.task not in models:
            raise ValueError("the name of model is not exits")
        return models[self.task][self.name](**self.param)


