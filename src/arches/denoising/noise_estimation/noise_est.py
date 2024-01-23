# -*- coding: utf-8 -*-
# @Time : 2024/01/23 13:49
# @Author : Liang Hao
# @FileName : noise_est.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from src.utils import ARCH_REGISTRY
from src.arches.denoising.spatial_channel.transformer import PSCT, SSCT

class ResidualGroup(nn.Module):
    def __init__(self, 
                 blk_name,
                 num_blk, 
                 in_dim, 
                 head_dim_s=32, 
                 head_num_c=4, 
                 winsize=8,
                 mlp_ratio_s=4, 
                 mlp_ratio_c=4,
                 drop_prob_s=0.,
                 drop_prob_c=0.,
                 bias=False,
                 pool=False,
                 pool_mode='conv',
                 ):
        
        if blk_name.lower() == 'ssct':
            block = SSCT
        else:
            block = PSCT
            
        module_body = []
        for index in range(num_blk):
            module_body.append(
                block(in_dim, head_dim_s=head_dim_s, head_num_c=head_num_c, 
                     winsize=winsize, index=index, mlp_ratio_s=mlp_ratio_s,
                     mlp_ratio_c=mlp_ratio_c, drop_prob_c=drop_prob_c, 
                     drop_prob_s=drop_prob_s, pool=pool, bias=bias,
                     pool_mode=pool_mode)
            )
        
        module_body.append(nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True))
        
        self.body = nn.Sequential(*module_body)
    
    def forward(self, x):
        return self.body(x) + x

@ARCH_REGISTRY.register()
class RIRNet(nn.Module):
    def __init__(self,
                 in_ch,
                 num_groups,
                 num_feats, 
                 head_dim_s=32, 
                 head_num_c=4, 
                 winsize=8,
                 mlp_ratio_s=4, 
                 mlp_ratio_c=4,
                 drop_rate=0.1,
                 bias=False,
                 pool=False,
                 pool_mode='conv',
                 blk_name='ssct'):
        super().__init__()
        
        module_head = [nn.Conv2d(in_ch, num_feats, 3, 1, 1, bias=True)]
        module_body = []
        
        drops = [x.item() for x in torch.linspace(0, drop_rate, sum(num_groups))]
        
        for i in range(len(num_groups)):
            num_blk = num_groups[i]
            module_body.append(
                ResidualGroup(num_blk=num_blk, in_dim=num_feats, head_dim_s=head_dim_s,
                              head_num_c=head_num_c, winsize=winsize,
                              mlp_ratio_c=mlp_ratio_c, mlp_ratio_s=mlp_ratio_s,
                              drop_prob_c=drops[i], drop_prob_s=drops[i],
                              bias=bias, pool=pool, pool_mode=pool_mode, blk_name=blk_name)
            )
        
        self.conv_out = nn.Conv2d(num_feats, num_feats, 3, 1, 1, bias=True)
        module_tail = [nn.Conv2d(num_feats, in_ch, 3, 1, 1, bias=True)]
        
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
    
    def forward(self, x):
        head = self.head(x)
        res = self.body(head)
        res = self.tail(self.conv_out(head + res)) + x
        return res