# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/13 17:02
# @File    :   official.py
# @Email   :   lianghao@whu.edu.cn

import torch.nn as nn
from .arch_utils import UNet_official
from einops.layers.torch import Rearrange
from source.net.basic_module.transformer.swin_transformer import SwinTransformerBlock as Official



class SwinUNetOfficial(UNet_official):
    def __init__(self, in_ch, num_feats, input_resolution, mid_blk_nums=1, enc_blk_nums=[],
                            dec_blk_nums=[], blk_name=None,blk_params=None):
        super(SwinUNetOfficial, self).__init__(in_ch, num_feats, input_resolution, mid_blk_nums, enc_blk_nums,
                                 dec_blk_nums, blk_name, blk_params)

    def __block__(self, blk_name):
        if blk_name.lower() == 'official':
            return SwinBlock_official


class SwinBlock_official(nn.Module):
    def __init__(self, num_feats, input_resolution, num_heads, window_size, type='W'):
        super(SwinBlock_official, self).__init__()
        shift_size = window_size // 2 if type == 'W' else 0
        self.size = input_resolution
        self.block = Official(num_feats, input_resolution, num_heads, window_size,
                              shift_size=shift_size)

        self.img2token = Rearrange('b c h w -> b (h w) c')
        self.token2img = Rearrange('b h w c -> b c h w')

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.img2token(x)
        x = self.block(x)
        x = x.reshape(b, h, w, c)
        x = self.token2img(x)
        return x

