# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/02 18:34:13
# @FileName:  arch_spatial.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
from .arch_utils import get_act_layer
from source.net.basic_module.utils.LayerNorm import LayerNorm2d

class _Concatention(nn.Module):
    def __init__(self, num_feats, ksizes=[3, 3, 3, 5, 3], act_layer=nn.ReLU(), bias=False, group=True):
        super(_Concatention, self).__init__()
        groups = num_feats if group else 1
        self.w_q = nn.Conv2d(num_feats, num_feats, kernel_size=ksizes[0], stride=1, padding=ksizes[0] // 2, bias=bias, groups=groups)
        self.w_k = nn.Conv2d(num_feats, num_feats, kernel_size=ksizes[1], stride=1, padding=ksizes[1] // 2, bias=bias, groups=groups)
        self.w_v = nn.Conv2d(num_feats, num_feats, kernel_size=ksizes[2], stride=1, padding=ksizes[2] // 2, bias=bias, groups=groups)
        
        self.concat_project = nn.Sequential(
            nn.Conv2d(num_feats * 2, num_feats, kernel_size=ksizes[3], stride=1, padding=ksizes[3] // 2, bias=bias, groups=groups),
            act_layer
        )
        
        self.w = nn.Conv2d(num_feats, num_feats, kernel_size=ksizes[4], stride=1, padding=ksizes[4] // 2, bias=bias, groups=groups)
    
    def forward(self, x):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        attn = self.concat_project(torch.cat([q, k], dim=1))
        x = self.w(attn * v)
        return x

class NonLocalTransformer(nn.Module):
    def __init__(self, num_feats, sa_expand, ffn_expand, ksizes, drop_rate=0.0, group=True, act_name_sa='relu', act_name_mlp='relu', bias=True):
        super(NonLocalTransformer, self).__init__()
        
        sa_feats = num_feats * sa_expand
        self.conv_1 = nn.Conv2d(num_feats, sa_feats, 1, 1, 0, groups=1, bias=True)
        self.act_layer_1 = get_act_layer(act_name_sa, sa_feats)
        self.attnetion = _Concatention(sa_feats, ksizes=ksizes, act_layer=self.act_layer_1, bias=bias, group=group)
        self.conv_2 = nn.Conv2d(sa_feats, num_feats, 1, 1, 0, groups=1, bias=True)
        
        ffn_feats = ffn_expand * num_feats
        self.mlp = self.get_mlp(num_feats, ffn_feats, act_name_mlp)
        
        self.norm_1 = LayerNorm2d(num_feats)
        self.norm_2 = LayerNorm2d(num_feats)
        
        self.dropout_1 = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
        self.dropout_2 = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros(size=(1, num_feats, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(size=(1, num_feats, 1, 1)), requires_grad=True)
    
    def forward(self, inp):
        x = inp
        x = self.attnetion(self.conv_1(self.norm_1(x)))
        x = self.dropout_1(self.conv_2(x))
        
        y = inp + x * self.beta
        
        x = self.dropout_2(self.mlp(self.norm_2(y)))
        return y + x * self.gamma
    
    
    def get_mlp(self, num_feats, ffn_feats, act_name):
        if act_name.lower() == 'gate':
            return nn.Sequential(
                nn.Conv2d(num_feats, ffn_feats, 1, 1, 0, groups=1, bias=True),
                get_act_layer('gate', ffn_feats),
                nn.Conv2d(ffn_feats, num_feats, 1, 1, 0, groups=1, bias=True)
            )
        elif act_name.lower() == 'sgate':
            return nn.Sequential(
                nn.Conv2d(num_feats, ffn_feats, 1, 1, 0, groups=1, bias=True),
                get_act_layer('sgate'),
                nn.Conv2d(ffn_feats // 2, num_feats, 1, 1, 0, groups=1, bias=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(num_feats, ffn_feats, 1, 1, 0, groups=1, bias=True),
                get_act_layer(act_name),
                nn.Conv2d(ffn_feats, num_feats, 1, 1, 0, groups=1, bias=True)
            )
    