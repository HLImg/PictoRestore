# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 15:33
# @File    :   standard.py
# @Email   :   lianghao@whu.edu.cn

import torch
from source.model.base_model import BaseModel
from source.utils.image.transpose import tensor2img

class standardModel(BaseModel):
    def __init__(self, config,  accelerator):
        super(standardModel, self).__init__(config,  accelerator)

    def __feed__(self, data):
        self.optimizer.zero_grad()
        predicted = self.net_g(data['lq'])
        loss = self.criterion(predicted, data['hq'])
        # ========================================= #
        self.accelerator.backward(loss)
        # ========================================= #
        self.optimizer.step()
        self.scheduler.step()

    def __eval__(self, data):
        predicted = self.net_g(data['lq'])
        res = {}
        for key, metric in self.metric:
            res[key] = metric(tensor2img(predicted),
                              tensor2img(data['hq']))

        return res
