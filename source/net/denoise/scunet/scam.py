# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/09 21:19:25
# @FileName:  scam.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn

from einops import rearrange
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange

class MCSA(nn.Module):
    def __init__(self, in_dim, out_dim, head_dim):
        super(MCSA, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_dim = head_dim
        
        self.project = nn.Linear(self.in_dim, self.in_dim * 3, bias=True)
        
        self.concat = nn.Linear(self.in_dim, self.out_dim)
        
    
    def forward(self, x):
        pass
        
        
        
        
        
        