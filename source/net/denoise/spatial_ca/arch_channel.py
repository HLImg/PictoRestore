# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/02 18:55:53
# @FileName:  arch_channel.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
from .arch_utils import get_act_layer
from source.net.basic_module.utils.LayerNorm import LayerNorm2d

class NAFBlock(nn.Module):
    def __init__(self, num_feats, sa_expand=2, ffn_expand=2, dropout_rate=0.):
        super(NAFBlock, self).__init__()
        
        sa_feats = num_feats * sa_expand
        self.conv_1 = nn.Conv2d(num_feats, sa_feats, 1, 1, 0, groups=1, bias=True)
        self.conv_2 = nn.Conv2d(sa_feats, sa_feats, 3, 1, 1, groups=num_feats, bias=True)
        self.conv_3 = nn.Conv2d(sa_feats // 2, num_feats, 1, 1, 0, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv2d(sa_expand // 2, sa_expand // 2, 1, 1, 0, bias=True, groups=1)
        )
        
        self.simplate_gate = get_act_layer('sgate')
        
        ffn_feats = ffn_expand * num_feats
        
        self.conv_4 = nn.Conv2d(num_feats, ffn_feats, 1, 1, 0, groups=1, bias=True)
        self.conv_5 = nn.Conv2d(ffn_feats // 2, num_feats, 1, 1, 0, groups=1, bias=True)
        
        self.norm_1 = LayerNorm2d(num_feats)
        self.norm_2 = LayerNorm2d(num_feats)
        
        self.dropout_1 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.dropout_2 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros(size=(1, num_feats, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(size=(1, num_feats, 1, 1)), requires_grad=True)
    
    def forward(self, inp):
        x = inp
        x = self.simplate_gate(self.conv_2(self.conv_1(self.norm_1(x))))
        x = self.dropout_1(self.conv_3(x * self.sca(x)))
        
        y = inp + x * self.beta
        
        x = self.simplate_gate(self.conv_4(self.norm_2(y)))
        x = self.dropout_2(self.conv_5(x))
        
        return y + x * self.gamma
    
        
        