# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/13 17:02
# @File    :   other.py
# @Email   :   lianghao@whu.edu.cn


import torch.nn as nn
from .arch_utils import UNet
from einops.layers.torch import Rearrange
from source.net.basic_module.transformer.swin_transformer_simple import SwinTransformerBlock as Other


class SwinUNet(UNet):
    def __init__(self, in_ch, num_feats, mid_blk_nums=1, enc_blk_nums=[],
                            dec_blk_nums=[], blk_name=None,blk_params=None):
        super(SwinUNet, self).__init__(in_ch, num_feats, mid_blk_nums, enc_blk_nums,
                                 dec_blk_nums, blk_name, blk_params)

    def __block__(self, blk_name):
        if blk_name.lower() == 'other':
            return SwinBlock_other


class SwinBlock_other(nn.Module):
    def __init__(self, num_feats, input_resolution, num_heads, window_size, type='W'):
        super(SwinBlock_other, self).__init__()
        self.block = Other(num_feats, num_feats, num_heads, window_size, drop_path=0.,
                           input_resolution=input_resolution, type=type)
        self.img2token = Rearrange('b c h w -> b h w c')
        self.token2img = Rearrange('b h w c -> b c h w')

    def forward(self, x):
        x = self.img2token(x)
        x = self.block(x)
        x = self.token2img(x)
        return x