# -*- coding: utf-8 -*-
# @Time    : 9/16/23 3:49 PM
# @File    : hsi.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch
import numpy as np
from source.model.base_model import BaseModel
from source.utils.image.transpose import tensor2img

class StandardModel(BaseModel):
    def __init__(self, config,  accelerator):
        super(hsiModel, self).__init__(config,  accelerator)

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
                res[key] = res.get(key, 0.) + metric(self.tensor2img(all_targets[ii]), 
                                                     self.tensor2img(all_predicts[ii]))
        return res

    def tensor2hsi(self, tensor, min_max=(0, 1)):
        _tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        img_np = _tensor.numpy()
        img_np = img_np.transpose(1, 2, 0)
        return img_np
    
    def tensor2img(self, tensor):
        return np.transpose(tensor.data.cpu().numpy().squeeze(), (1, 2, 0))
