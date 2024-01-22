# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:20
# @Author : Liang Hao
# @FileName : transform.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from einops import rearrange
from .utils import LayerNorm2d
from .utils import WMSA as SpatialSA
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_


class ChannelSA(nn.Module):
    def __init__(self, dim, num_heads, bias, pool=False, pool_mode='conv'):
        super().__init__()
        
        self.num_heads = num_heads
        self.project = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim * 3)
        )
        
        if pool:
            if pool_mode.lower() == 'conv':
                self.down_sample = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, bias=False, groups=dim * 2)
            elif pool_mode.lower() == 'pool':
                self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise ValueError(f"pool_mode must be conv and pool")
        else:
            self.down_sample = nn.Identity()
        
        self.pos_emb = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.to_multi = Rearrange('b (head c) h w -> b head c (h w)', head=num_heads)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.project(x)
        q, k, v = qkv.chunk(3, dim=1)
        qk = torch.cat([q, k], dim=1)
        qk = self.down_sample(qk)
        q, k = qk.chunk(2, dim=1)
        '''
        q : b c nh nw ===> b head c L
        k : b c nh nw ===> b head c L
        v : b c h w ===> b head c hw
        '''
        
        q, k, v = self.to_multi(q), self.to_multi(k), self.to_multi(v)
            
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # b head c c
        d = q.shape[-1]
        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5) + self.pos_emb
        attn = attn.softmax(dim=-1)
        
        # b head c c @ b head c hw => b head c hw
        out = attn @ v
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        out = self.project_out(out)
        return out
    

class ChannelTransformer(nn.Module):
    def __init__(self, in_dim, num_heads=4, drop_prob=0., mlp_ratio=4, bias=False, pool=False, pool_mode='conv'):
        super().__init__()
        
        self.norm_1 = LayerNorm2d(in_dim)
        self.msa = ChannelSA(
            dim=in_dim, num_heads=num_heads, bias=bias, pool=pool, pool_mode=pool_mode
        )
        
        self.norm_2 = LayerNorm2d(in_dim)
        
        mlp_dim = in_dim * mlp_ratio
        
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
            nn.Conv2d(mlp_dim, mlp_dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=mlp_dim),
            nn.GELU(),
            nn.Conv2d(mlp_dim, in_dim, kernel_size=1, padding=0, bias=bias)
        )
        
        self.drop_path = DropPath(drop_prob=drop_prob) if drop_prob > 0. else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.msa(self.norm_1(x)))
        x = x + self.drop_path(self.ffn(self.norm_2(x)))
        return x
    

class SpatialTransformer(nn.Module):
    def __init__(self, in_dim, mlp_ratio, head_dim, winsize, drop_prob=0., type='w'):
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(in_dim)
        self.msa = SpatialSA(
            in_dim, in_dim, head_dim=head_dim, window_size=winsize, type=type
        )
        
        self.drop_path = DropPath(drop_prob=drop_prob) if drop_prob > 0. else nn.Identity()
        
        self.norm_2 = nn.LayerNorm(in_dim)
        
        mlp_dim = in_dim * mlp_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, in_dim)
        )
        
        
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = x + self.drop_path(self.msa(self.norm_1(x)))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        x = rearrange(x, 'b h w c -> b c h w')
        return x
    

class SSCT(nn.Module):
    # Series Spatail and Channel Transformer
    def __init__(self, 
                 in_dim, 
                 head_dim_s=32, 
                 head_num_c=4, 
                 winsize=8, 
                 index = 1,
                 mlp_ratio_s=4, 
                 mlp_ratio_c=4,
                 drop_prob_s=0.,
                 drop_prob_c=0.,
                 bias=False,
                 pool=False,
                 pool_mode='conv', 
                 **kwargs):
        super().__init__()
        
        swin_type = 'SW' if (index + 1) % 2 == 0 else 'W'

        self.spatial = SpatialTransformer(
            in_dim=in_dim, mlp_ratio=mlp_ratio_s, head_dim=head_dim_s,
            winsize=winsize, drop_prob=drop_prob_s, type=swin_type
        )
        
        self.channel = ChannelTransformer(
            in_dim=in_dim, num_heads=head_num_c, drop_prob=drop_prob_c,
            mlp_ratio=mlp_ratio_c, bias=bias, pool=pool, pool_mode=pool_mode
        )
        
        self.beta = nn.Parameter(torch.zeros((1, in_dim, 1, 1)), requires_grad=True)
    
    def forward(self, x):
        f = self.channel(self.spatial(x))
        return f + self.beta * x
        
        
class PSCT(nn.Module):
    #Parallel Spatail and Channel Transformer
    def __init__(self, 
                 in_dim, 
                 head_dim_s=32, 
                 head_num_c=4, 
                 winsize=8, 
                 index = 1,
                 mlp_ratio_s=4, 
                 mlp_ratio_c=4,
                 drop_prob_s=0.,
                 drop_prob_c=0.,
                 bias=False,
                 pool=False,
                 pool_mode='conv',
                 **kwargs):
        super().__init__()

        swin_type = 'SW' if (index + 1) % 2 == 0 else 'W'

        self.conv_split = nn.Conv2d(in_dim, in_dim * 2, kernel_size=1, padding=0,  bias=True)
        self.conv_fusion = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, padding=0, bias=False)
        
        self.spatial = SpatialTransformer(
            in_dim=in_dim, mlp_ratio=mlp_ratio_s, head_dim=head_dim_s,
            winsize=winsize, drop_prob=drop_prob_s, type=swin_type
        )
        
        self.channel = ChannelTransformer(
            in_dim=in_dim, num_heads=head_num_c, drop_prob=drop_prob_c,
            mlp_ratio=mlp_ratio_c, bias=bias, pool=pool, pool_mode=pool_mode
        )
        
        self.beta = nn.Parameter(torch.zeros((1, in_dim, 1, 1)), requires_grad=True)
        
    
    def forward(self, x):
        f = self.conv_split(x)
        f1, f2 = f.chunk(2, dim=1)
        f1 = self.spatial(f1)
        f2 = self.channel(f2)
        f = self.conv_fusion(torch.cat([f1, f2], dim=1))
        
        return f + self.beta * x
        
        