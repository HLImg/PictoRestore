# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/01 12:34:46
# @FileName:  nonlocal_ca.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
from .arch_utils import NonLocalTransformer
from source.net.denoise.rgb.nafnet import NAFBlock

class NonLocalCA(nn.Module):
    def __init__(self, in_ch, expand_embed, expand_mlp, ksizes, act_name, bias, group, score_funtion='concat', drop_path=0.0):
        super(NonLocalCA, self).__init__()
        
        self.sa_block = NonLocalTransformer(in_ch, expand_embed, expand_mlp, ksizes, act_name, bias, group, score_funtion='concat', drop_path=0.0)
        
        self.ca_block = NAFBlock(in_ch, DW_Expand=expand_embed, FFN_Expand=expand_mlp, drop_out_rate=drop_path)
    
    
    def forward(self, x):
        x = self.sa_block(x)
        x = self.ca_block(x)
        return x
        