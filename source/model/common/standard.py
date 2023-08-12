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
        self.loss = loss.item() 
        # ========================================= #
        self.accelerator.backward(loss)
        # ========================================= #
        self.optimizer.step()
        self.scheduler.step()

    def __eval__(self, data):
        predicted = self.net_g(data['lq'])
        # ======================================== #
        all_predicts, all_targets = self.accelerator.gather_for_metrics((predicted, data['hq']))
        # ======================================== #
        res = {}
        for key, metric in self.metric.items():
            for ii in range(all_predicts.shape[0]):
                res[key] = res.get(key, 0.) + metric(tensor2img(all_predicts[ii]),
                                                     tensor2img(all_targets[ii]))
        return res
