# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/07 20:43:04
# @FileName:  non_local.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
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

class NonLocalTransformerv1(nn.Module):
    def __init__(self, num_feats, ffn_expand, ksizes, drop_rate=0.0, bias=True, group=True):
        super(NonLocalTransformerv1, self).__init__()
        print("drop_path_rate:{:.6f}".format(drop_rate))
        self.ln1 = LayerNorm2d(num_feats)
        self.rcsa = _Concatention(num_feats, ksizes, act_layer=nn.ReLU(), bias=bias, group=group)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.ln2 = LayerNorm2d(num_feats)
        # channel mixing
        self.mlp = nn.Sequential(
            nn.Linear(num_feats, num_feats * ffn_expand),
            nn.GELU(),
            nn.Linear(num_feats * ffn_expand, num_feats)
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.rcsa(self.ln1(x)))
        y = Rearrange('b c h w -> b h w c')(self.ln2(x))
        y = self.drop_path(self.mlp(y))
        x = x + Rearrange('b h w c -> b c h w')(y)
        return x

class ConvNonLocalBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, ksizes, ffn_expand, drop_rate=0., bias=False, group=True):
        super(ConvNonLocalBlock, self).__init__()
        
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        
        self.trans_block = NonLocalTransformerv1(trans_dim, ffn_expand, ksizes, drop_rate=drop_rate, bias=bias, group=group)
        
        self.conv_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.trans_dim + self.conv_dim, 1, 1, 0, bias=True)
        self.conv_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.trans_dim + self.conv_dim, 1, 1, 0, bias=True)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )
    
    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv_1(x), (self.conv_dim, self.trans_dim), dim=1)
        
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        
        res = self.conv_2(torch.cat((conv_x, trans_x), dim=1))
        
        x = x + res
        return x
            