# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/24 13:13:45
# @FileName:  non_local.py
# @Contact :  lianghao02@megvii.com

import torch
import torch.nn as nn

class NonLocalAttention(nn.Module):
    def __init__(self, in_ch, expand_dim, ksizes, act_name, bias, group, score_funtion='concat'):
        super(NonLocalAttention, self).__init__()
        num_feats = in_ch * expand_dim
        groups = num_feats if group else 1
        act_layer = self.get_act_layer(act_name)
        if score_funtion.lower() == 'concat':
            self.sa = _Concatention(in_ch, num_feats, ksizes=ksizes, act_layer=act_layer, bias=bias, groups=groups)
        else:
            raise "the score function is not exist"
        
    def forward(self, x):
        return self.sa(x)
    
    
    def get_act_layer(self, name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'softmax':
            return nn.Softmax2d()
        else:
            raise "not exits in get act layer"
    

class _Concatention(nn.Module):
    def __init__(self, in_ch, num_feats, ksizes=[3, 3, 3, 5], act_layer=nn.ReLU(), bias=False, groups=1):
        super(_Concatention, self).__init__()
        
        self.w_q = nn.Conv2d(in_ch, num_feats, kernel_size=ksizes[0], 
                             stride=1, padding=ksizes[0] // 2, bias=bias, groups=groups)
        self.w_k = nn.Conv2d(in_ch, num_feats, kernel_size=ksizes[1], 
                             stride=1, padding=ksizes[1] // 2, bias=bias, groups=groups)
        self.w_v = nn.Conv2d(in_ch, num_feats, kernel_size=ksizes[2], 
                             stride=1, padding=ksizes[2] // 2, bias=bias, groups=groups)
        
        
        self.concat_project = nn.Sequential(
            nn.Conv2d(num_feats * 2, num_feats, kernel_size=ksizes[3], 
                      stride=1, padding=ksizes[3] // 2, bias=False, groups=groups), 
            act_layer
        )
        
        self.w = nn.Conv2d(num_feats, in_ch, kernel_size=ksizes[3], 
                           stride=1, padding=ksizes[3] // 2, bias=False, groups=groups)
    
    def forward(self, x):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        attn = self.concat_project(torch.cat([q, k], dim=1))
        x = self.w(attn * v)
        return x
        

        