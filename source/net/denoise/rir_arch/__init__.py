# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/05 22:14:48
# @FileName:  __init__.py
# @Contact :  lianghao@whu.edu.cn

import torch
from source.net.backbone.rir_arch import RIR
from source.net.denoise.rgb.nafnet import NAFBlock

class RIRDenoising(RIR):
    def __init__(self, in_ch, num_feats, kernel_size, num_group=..., blk_name=None, blk_params=None, bias=True):
        super(RIRDenoising, self).__init__(in_ch, num_feats, kernel_size, num_group, blk_name, blk_params, bias)
    
    def __block__(self, blk_name):
        if blk_name.lower() == 'nafnet':
            return NAFBlock
        else:
            raise 'blk_name is not exist'