# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/23 16:26:09
# @FileName:  arch_utils.py
# @Contact :  lianghao02@megvii.com

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from source.net.basic_module.utils import LayerNorm2d
from source.net.basic_module.transformer.swin_transformer_simple import WMSA

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super(EnhancedSpatialAttention, self).__init__()
        
        f = in_ch // 4
        num_feature = in_ch
        self.conv1 = nn.Conv2d(in_channels=num_feature, out_channels=f, kernel_size=1)
        self.conv_f = nn.Conv2d(in_channels=f, out_channels=f, kernel_size=1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = nn.Conv2d(f, f, 3, 1, 1)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = nn.Conv2d(f, f, 3, 1, 1)
        self.conv3_ = nn.Conv2d(f, f, 3, 1, 1)
        self.conv4 = nn.Conv2d(f, in_ch, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()
    
    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        out = x * m
        return out


class ResSwinTCBlock(nn.Module):
    def __init__(self, in_ch, split_dim, head_dim, window_size, drop_path, type_='W', input_resolution=None):
        super(ResSwinTCBlock, self).__init__()
        assert sum(split_dim) == in_ch, f'split dim : {split_dim}, but sum_dim : {in_ch}'
        self.conv_dim, self.trans_dim = split_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if input_resolution <= window_size:
            type_ = 'W'
        
        """window self-attention wth enhanced spatial attention"""
        self.ln_1 = LayerNorm2d(self.trans_dim)
        self.msa = WMSA(self.trans_dim, self.trans_dim, head_dim, window_size, type=type_)
        self.ln_2 = LayerNorm2d(self.trans_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.trans_dim, self.trans_dim * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(self.trans_dim * 4, self.trans_dim * 4, 3, 1, 1, groups=self.trans_dim * 4),
            nn.GELU(),
            nn.Conv2d(self.trans_dim * 4, self.trans_dim, 1, 1, 0)
        )
        
        self.sw_esa = EnhancedSpatialAttention(self.trans_dim)
        
        """residual convolution block"""
        self.conv_res = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )
        self.conv_res_path = nn.Conv2d(self.conv_dim, self.conv_dim, 1, 1, 0, bias=True)
        self.conv_end = nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=True)
        
        self.chw2hwc = Rearrange('b c h w -> b h w c')
        self.hwc2chw = Rearrange('b h w c -> b c h w')
        
    
    def forward(self, x):
        x_conv, x_trans = torch.split(x, (self.conv_dim, self.trans_dim), dim=1)
        """window self-attention"""
        y = self.hwc2chw(self.msa(self.chw2hwc(self.ln_1(x_trans))))
        x_trans = x_trans + self.drop_path(y)
        x_trans = x_trans + self.drop_path(self.mlp(self.ln_2(x_trans)))
        x_trans = self.sw_esa(x_trans)
        
        """residual convolution"""
        x_conv = self.conv_res_path(x_conv) + self.drop_path(self.conv_res(x_conv))
        x = x + self.conv_end(torch.cat([x_conv, x_trans], dim=1))

        return x