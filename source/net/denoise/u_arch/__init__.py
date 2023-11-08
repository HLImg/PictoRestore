# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/05 22:16:39
# @FileName:  __init__.py
# @Contact :  lianghao@whu.edu.cn

import torch.nn as nn
from source.net.backbone.unet_arch import UNet
from source.net.denoise.rgb.nafnet import NAFBlock


class UnetDenoising(UNet):
    def __init__(self, in_ch, num_feats, mid_blk_nums=1, enc_blk_nums=..., dec_blk_nums=..., blk_name=None, blk_params=None):
        super(UnetDenoising, self).__init__(in_ch, num_feats, mid_blk_nums, enc_blk_nums, dec_blk_nums, blk_name, blk_params)
        
    def __block__(self, blk_name):
        if blk_name.lower() == "nafnet":
            return NAFBlock
        else:
            raise "the blk_name is not exist"