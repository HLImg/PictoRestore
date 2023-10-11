# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/10 20:17:06
# @FileName:  swin_restormer.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .tcb import WMSA

from einops import rearrange
from timm.models.layers import DropPath
from source.net.basic_module.utils.LayerNorm import LayerNorm2d

def get_act_layer(act_name='relu'):
        if act_name.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_name.lower() == 'gelu':
            return nn.GELU()
        else:
            raise "incorrect act name"

class CAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAttention, self).__init__()
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(NonLocalBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
        

class LocalBlock(nn.Module):
    def __init__(self,  conv_dim, num_heads, act_name='relu', is_group=True, is_bias=True, drop_rate=0.):
        super(LocalBlock, self).__init__()
        
        self.norm_1 = LayerNorm2d(conv_dim)
        self.local_spatial = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=is_bias, groups=conv_dim if is_group else 1),
            get_act_layer(act_name),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=is_bias, groups=conv_dim if is_group else 1)
        )
        
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        
        self.norm_2 = LayerNorm2d(conv_dim)
        self.cattn = CAttention(conv_dim, num_heads, bias=is_bias)
        
    def forward(self, x):
        x = x + self.drop_path(self.local_spatial(self.norm_1(x)))
        x = x + self.drop_path(self.cattn(self.norm_2(x)))
        return x
    

class CATransBlock(nn.Module):
    def __init__(self, conv_dim, conv_num_head, trans_dim, 
                 trans_head_dim, window_size, type, input_resolution,
                 act_conv='relu', drop_trans=0., drop_conv=0.):
        super(CATransBlock, self).__init__()
        
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        
        assert type in ['W', 'SW']
        
        if input_resolution <= window_size:
            type = 'W'
        
        self.conv_head = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        
        self.nonlocal_block = NonLocalBlock(trans_dim, trans_dim, trans_head_dim, window_size, 
                                            drop_path=drop_trans, type=type, 
                                            input_resolution=input_resolution)
        
        self.local_block = LocalBlock(conv_dim, num_heads=conv_num_head, 
                                      act_name=act_conv, drop_rate=drop_conv)
        
        self.conv_tail = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
    
    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv_head(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.local_block(conv_x)
        
        trans_x = rearrange(trans_x, 'b c h w -> b h w c')
        trans_x = self.nonlocal_block(trans_x)
        trans_x = rearrange(trans_x, 'b h w c -> b c h w')
        
        res = self.conv_tail(torch.cat((conv_x, trans_x), dim=1)) 
        
        x = x + res
        
        return x
    
    

class ResidualGroup(nn.Module):
    def __init__(self, num_blk, conv_dim, trans_dim, drop_conv, drop_trans, act_conv,
                 conv_num_head, trans_head_dim, window_size, input_resolution):
        super(ResidualGroup, self).__init__()    
        
        module_body = [
            CATransBlock(conv_dim, conv_num_head, trans_dim, trans_head_dim, 
                         window_size, type='W' if not i%2 else 'SW', 
                         input_resolution=input_resolution, act_conv=act_conv, 
                         drop_conv=drop_conv[i], drop_trans=drop_trans[i]
                         )
            for i in range(num_blk)
        ]  
        
        module_body.append(nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 3, 1, 1, bias=True))
        
        self.body = nn.Sequential(*module_body)
    
    def forward(self, x):
        return self.body(x) + x


class RIRCANet(nn.Module):
    def __init__(self, in_ch, conv_dim, trans_dim, drop_conv, drop_trans, 
                 act_conv, num_groups, conv_num_head, trans_head_dim, window_size, input_resolution=None):
        super(RIRCANet, self).__init__()
        
        num_feats = conv_dim + trans_dim
        module_head = [nn.Conv2d(in_ch, num_feats, 3, 1, 1, bias=True)]
        
        module_body = []
        drops_conv = [x.item() for x in torch.linspace(0, drop_conv, sum(num_groups))]
        drops_trans = [x.item() for x in torch.linspace(0, drop_trans, sum(num_groups))]
        for i in range(len(num_groups)):
            num_blk = num_groups[i]
            module_body.append(
                ResidualGroup(num_blk, conv_dim, trans_dim, drop_conv=drop_conv[i : i + num_blk], 
                              drop_trans=drop_trans[i : i + num_blk], act_conv=act_conv, 
                              conv_num_head=conv_num_head, trans_head_dim=trans_head_dim, window_size=window_size, 
                              input_resolution=input_resolution
                )
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