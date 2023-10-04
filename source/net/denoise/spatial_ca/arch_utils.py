# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/02 18:34:49
# @FileName:  arch_utils.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, num_feats):
        super(Gate, self).__init__()
        self.conv = nn.Conv2d(num_feats, num_feats * 2, 1, 1, 0, bias=False, groups=1)
    
    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def get_act_layer(act_name, num_feats=10):
    if act_name.lower() == 'relu':
        return nn.ReLU()
    elif act_name.lower() == 'gelu':
        return nn.GELU()
    elif act_name.lower() == 'gate':
        return Gate(num_feats=num_feats)
    elif act_name.lower() == 'sgate':
        return SimpleGate()
    else:
        raise "the act_layer is not exits"
    
if __name__ == '__main__':
    act = get_act_layer('relu', {})
    a = torch.randn(1, 3, 5, 5)
    b = act(a)
    print(b.shape)