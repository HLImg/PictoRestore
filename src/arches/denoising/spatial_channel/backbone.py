# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:19
# @Author : Liang Hao
# @FileName : backbone.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from .utils import Downsample, UpSample
from .transformer import PSCT, SSCT
from src.utils import ARCH_REGISTRY

class BaseUnet(nn.Module):
    def __init__(self,
                 in_ch, 
                 num_feats, 
                 num_enc_blk=[2, 2], 
                 num_mid_blk=2, 
                 num_dec_blk=[2, 2],
                 blk_name=None,
                 blk_param=None):
        super(BaseUnet, self).__init__()
        block = self.__block__(blk_name)
        
        self.in_project = nn.Conv2d(
            in_ch, num_feats, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pad_size = 2 ** len(num_enc_blk)
        
        self.out_project = nn.Conv2d(
            num_feats, in_ch, kernel_size=3, padding=1, stride=1, bias=True
        )
        
        # encoder with downsampling
        wf = num_feats
        for num in num_enc_blk:
            self.encoders.append(nn.Sequential(
                *[block(wf, index=index, **blk_param) for index  in range(num)]
            ))
            self.downs.append(Downsample(wf))
            wf = wf * 2
        
        # middle 
        self.middle_blks = nn.Sequential(
            *[block(wf, index=index, **blk_param) for index in range(num_mid_blk)]
        )
        
        # decoder with upsampling
        for num in num_dec_blk:
            self.ups.append(UpSample(wf))
            wf = wf // 2
            self.decoders.append(nn.Sequential(
                *[block(wf, index=index, **blk_param) for index  in range(num)]
            ))
            
    
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
    
        encs = []
        x = self.in_project(inp)
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.out_project(x)
        x = x + inp
        
        return x[:, :, :H, :W]  
    
    def check_image_size(self, x):
        _, _, h, w = x.shape
        ph = (self.pad_size - h % self.pad_size) % self.pad_size
        pw = (self.pad_size - w % self.pad_size) % self.pad_size
        x = F.pad(x, (0, pw, 0, ph))
        return x
    
    @abstractmethod
    def __block__(blk_name):
        pass        
    
@ARCH_REGISTRY.register()
class SCTUnet(BaseUnet):
    def __init__(self, in_ch, num_feats, num_enc_blk=[2, 2], num_mid_blk=2, num_dec_blk=[2, 2], blk_name=None, blk_param=None):
        super(SCTUnet, self).__init__(in_ch, num_feats, num_enc_blk, num_mid_blk, num_dec_blk, blk_name, blk_param)
        
    
    def __block__(self, blk_name):
        if blk_name.lower() == 'psct':
            return PSCT
        elif blk_name.lower() == 'ssct':
            return SSCT
        