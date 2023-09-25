# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/23 16:25:13
# @FileName:  __init__.py
# @Contact :  lianghao02@megvii.com

import torch
from source.net.backbone.rir_arch import RIR
from .arch_utils import ResSwinTCBlock, NonLocalTransformer


class TCNet(RIR):
    def __init__(self, in_ch, num_feats, kernel_size, num_group=..., blk_name=None, blk_params=None, bias=True):
        super(TCNet, self).__init__(in_ch, num_feats, kernel_size, num_group, blk_name, blk_params, bias)
    
    def __block__(self, blk_name):
        if blk_name.lower() == 'tcnet':
            print(blk_name)
            return ResSwinTCBlock
        elif blk_name.lower() == 'nonlocal':
            return NonLocalTransformer
        else:
            raise "blk_name is not exist"


if __name__ == '__main__':
    blk_param = {
        'head_dim' : 32,
        'window_size': 8, 
        'drop_path' : 0., 
        'type_': 'W',
        'input_resolution': 224
    }
    
    x = torch.randn(1, 3, 128, 128)
    tcnet = TCNet(3, 32, 3, num_group=[2, 2, 2], blk_name='tcnet', blk_params=blk_param, bias=False)
    y = tcnet(x)
    print(x.shape, y.shape)
    