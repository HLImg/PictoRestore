# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/08 14:30:23
# @FileName:  scanet.py
# @Contact :  lianghao@whu.edu.cn


import torch
import numpy as np
import torch.nn as nn

from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from source.net.basic_module.utils.LayerNorm import LayerNorm2d

def get_act_layer(act_name='relu'):
        if act_name.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_name.lower() == 'gelu':
            return nn.GELU()
        else:
            raise "incorrect act name"

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class NonLocalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None, act_name='gelu'):
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
            get_act_layer(act_name),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class LocalBlock(nn.Module):
    def __init__(self, conv_dim, drop_path=0., act_name='relu', ca_expand=2, is_group=True, is_bias=True):
        super(LocalBlock, self).__init__()
        
        self.norm_1 = LayerNorm2d(conv_dim)
        self.local_spatial = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=is_bias, groups=conv_dim if is_group else 1),
            get_act_layer(act_name),
            nn.Conv1d(conv_dim, conv_dim, 3, 1, 1, bias=is_bias, groups=conv_dim if is_group else 1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm_2 = LayerNorm2d(conv_dim)
        expand_dim = conv_dim * ca_expand
        self.ch_attn = nn.Sequential(
            nn.Conv2d(conv_dim, expand_dim, 1, 1, 0, bias=is_bias),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expand_dim, conv_dim, 1, 1, 0, bias=is_bias)
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.local_spatial(self.norm_1(x)))
        x = x + self.drop_path(x * self.ch_attn(x))
        return x
    

class CATransBlock(nn.Module):
    def __init__(self, conv_dim, drop_conv, act_conv, ca_expand, is_group, is_bias,
                 trans_dim, head_dim, window_size, drop_trans, win_type, input_resolution=None, act_trans='gelu'):
        super(CATransBlock, self).__init__()
        
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        
        assert win_type in ['W', 'SW']
        
        if input_resolution <= window_size:
            win_type = 'W'
        
        self.conv_head = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        
        self.nonlocal_block = NonLocalBlock(trans_dim, trans_dim, head_dim, 
                                            window_size, drop_path=drop_trans, type=window_size, 
                                            input_resolution=input_resolution, act_name=act_trans)
        
        self.local_block = LocalBlock(conv_dim, drop_path=drop_conv, 
                                      act_name=act_conv, ca_expand=ca_expand,
                                      is_group=is_group, is_bias=is_bias)
        
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
    def __init__(self, num_blk, conv_dim, drop_conv, act_conv, ca_expand, is_group, is_bias,
                 trans_dim, head_dim, window_size, drop_trans, input_resolution=None, act_trans='gelu'):
        super(ResidualGroup, self).__init__()
        
        module_body = [
            CATransBlock(conv_dim, drop_conv=drop_conv[i], act_conv=act_conv, ca_expand=ca_expand,
                         is_group=is_group, is_bias=is_bias, trans_dim=trans_dim, head_dim=head_dim,
                         window_size=window_size, drop_trans=drop_trans[i], win_type='W' if not i%2 else 'SW', 
                         input_resolution=input_resolution, act_trans=act_trans)
                for i in range(num_blk)
        ]
        
        module_body.append(nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 3, 1, 1, bias=True))
        
        self.body = nn.Sequential(*module_body)
    
    def forward(self, x):
        return self.body(x) + x
        
    
class RIRSCANet(nn.Module):
    def __init__(self, in_ch, conv_dim, drop_conv, act_conv, num_groups, ca_expand, is_group, is_bias,
                 trans_dim, head_dim, window_size, drop_trans, input_resolution=None, act_trans='gelu'):
        super(RIRSCANet, self).__init__()
        
        num_feats = conv_dim + trans_dim
        module_head = [nn.Conv2d(in_ch, num_feats, 3, 1, 1, bias=True)]
        module_body = []
        
        drops_conv = [x.item() for x in torch.linspace(0, drop_conv, sum(num_groups))]
        drops_trans = [x.item() for x in torch.linspace(0, drop_trans, sum(num_groups))]
        
        for i in range(len(num_groups)):
            num_blk = num_groups[i]
            module_body.append(
                ResidualGroup(num_blk, conv_dim, drop_conv=drops_conv[i : i + num_blk], 
                              act_conv=act_conv, ca_expand=ca_expand, is_group=is_group, 
                              is_bias=is_bias, trans_dim=trans_dim, head_dim=head_dim, 
                              window_size=window_size, drop_trans=drops_trans[i : i + num_blk],
                              input_resolution=input_resolution, act_trans=act_trans)
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