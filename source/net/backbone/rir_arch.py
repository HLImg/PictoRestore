# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/23 15:40:24
# @FileName:  rir_arch.py
# @Contact :  lianghao02@megvii.com

import torch
import torch.nn as nn
import torch.nn.functional as F

class RIR(nn.Module):
    def __init__(self, in_ch, num_feats, kernel_size, num_group=[], blk_name=None, blk_params=None, bias=True):
        super(RIR, self).__init__()
        blk = self.__block__(blk_name)
        
        drop_rate = blk_params.get('drop_rate', 0)
        drops = [x.item() for x in torch.linspace(0, drop_rate, sum(num_group))]
        
        module_head = [nn.Conv2d(in_ch, num_feats, kernel_size, 1, padding=kernel_size//2)]
        module_body = []
        for i in range(len(num_group)):
            module_body.append(
                ResidualGroup(num_feats, drop_rate=drops[i : i + num_group[i]], blk_nums=num_group[i], blk=blk, blk_params=blk_params)
            )
            
        self.conv_out = nn.Conv2d(num_feats, num_feats, kernel_size, 1, padding=kernel_size // 2, bias=bias)
        
        module_tail = [nn.Conv2d(num_feats, in_ch, 3, 1, 1, bias=True)]
        
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
    
    def forward(self, x):
        head = self.head(x)
        res = self.body(head)
        res = self.tail(self.conv_out(res + head)) + x
        return res
    
    def __block__(self, blk_name):
        pass    

    
class ResidualGroup(nn.Module):
    def __init__(self, num_feats, drop_rate, blk_nums=10, blk=None, blk_params=None):
        super(ResidualGroup, self).__init__()
        
        if 'drop_rate' in blk_params.keys():
            blk_params['drop_rate'] = drop_rate
            
        module_body = []
        
        for i in range(blk_nums):
            if 'drop_rate' in blk_params.keys():
                blk_params['drop_rate'] = drop_rate[i]
            module_body.append( blk(num_feats, **blk_params))
        
        module_body.append(nn.Conv2d(num_feats, num_feats, 3, 1, 1))
        self.body = nn.Sequential(*module_body)
    
    def forward(self, x):
        res = self.body(x) + x
        return res
        
    
        