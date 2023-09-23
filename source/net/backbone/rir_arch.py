# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/23 15:40:24
# @FileName:  rir_arch.py
# @Contact :  lianghao02@megvii.com

import torch.nn as nn
import torch.nn.functional as F

class RIR(nn.Module):
    def __init__(self, in_ch, num_feats, kernel_size, num_group=[], blk_name=None, blk_params=None, bias=True):
        super(RIR, self).__init__()
        blk = self.__block__(blk_name)
        module_head = [nn.Conv2d(in_ch, num_feats, kernel_size, 1, padding=kernel_size//2, bias=bias)]
        module_body = []
        for blk_num in num_group:
            module_body.append(
                ResidualGroup(num_feats, blk_nums=blk_num, blk=blk, blk_name=blk_name, blk_params=blk_params, bias=bias)
            )
        module_body.append(nn.Conv2d(num_feats, num_feats, kernel_size, 1, padding=kernel_size // 2, bias=bias))
        module_tail = [nn.Conv2d(num_feats, in_ch, kernel_size, 1, 1, bias=False)]
        
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
    
    def forward(self, x):
        head = self.head(x)
        res = self.body(head) + head
        res = self.tail(res) + x
        return res
    
    def __block__(self, blk_name):
        pass    

    
class ResidualGroup(nn.Module):
    def __init__(self, num_feats, blk_nums=10, blk=None, blk_name=None, blk_params=None, bias=True):
        super(ResidualGroup, self).__init__()
        module_body = [
            blk(num_feats, **blk_params) for _ in range(blk_nums)
        ]
        module_body.append(nn.Conv2d(num_feats, num_feats, 3, 1, 1, bias=bias))
        self.body = nn.Sequential(*module_body)
    
    def forward(self, x):
        res = self.body(x) + x
        return res
        
    
        