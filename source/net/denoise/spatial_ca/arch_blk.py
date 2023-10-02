# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/02 20:24:54
# @FileName:  arch_blk.py
# @Contact :  lianghao@whu.edu.cn

import torch.nn as nn

from .arch_spatial import NonLocalTransformer
from source.net.denoise.nafnet.net_utils import NAFBlock

class NonLocalCAV2(nn.Module):
    def __init__(self, in_ch, sa_expand, ffn_expand, ksizes, dropout_rate=0.0, group=True, act_name_sa='relu', act_name_mlp='relu', bias=True):
        super(NonLocalCAV2, self).__init__()
        
        self.sa_block = NonLocalTransformer(in_ch, sa_expand, ffn_expand, ksizes, dropout_rate, 
                                            group, act_name_sa, act_name_mlp, bias)
        self.ca_block = NAFBlock(in_ch, sa_expand, ffn_expand, dropout_rate)
        
    def forward(self, x):
        x = self.sa_block(x)
        x = self.ca_block(x)
        return x