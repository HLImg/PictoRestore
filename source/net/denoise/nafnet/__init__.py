# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:48
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .net_utils import NAFBlock
from source.net.backbone.unet_arch import UNet
from source.net.denoise.swinconv.nonlocal_ca import NonLocalCA
from source.net.denoise.spatial_ca.arch_blk import NonLocalCAV2
from source.net.denoise.swinconv.arch_utils import NonLocalTransformer, ResSwinTCBlock


class NAFNet(UNet):
    def __init__(self, in_ch, num_feats, mid_blk_nums=1, enc_blk_nums=[], dec_blk_nums=[], blk_name=None,
                 blk_params=None):
        super(NAFNet, self).__init__(in_ch, num_feats, mid_blk_nums, enc_blk_nums, dec_blk_nums, blk_name, blk_params)

    def __block__(self, blk_name):
        if blk_name.lower() == "nafnet":
            return NAFBlock
        elif blk_name.lower() == 'tcnet':
            return ResSwinTCBlock
        elif blk_name.lower() == 'nonlocal':
            return NonLocalTransformer
        elif blk_name.lower() == 'nonlocal_ca':
            return NonLocalCA
        elif blk_name.lower() == 'nonlocal_ca_v2':
            return NonLocalCAV2
        else:
            raise "the blk_name is not exist"