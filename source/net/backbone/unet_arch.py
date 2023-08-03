# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 13:40
# @File    :   unet_arch.py
# @Email   :   lianghao@whu.edu.cn


import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_ch, num_feats, mid_blk_nums=1, enc_blk_nums=[], dec_blk_nums=[], blk_name=None, blk_params=None):
        super(UNet, self).__init__()

        blk = self.__block__(blk_name)

        self.intro = nn.Conv2d(in_ch, num_feats, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.ending = nn.Conv2d(num_feats, in_ch, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        wf = num_feats
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[blk(wf, **blk_params) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(wf, 2 * wf, 2, 2)
            )
            wf = wf * 2

        self.middle_blks = nn.Sequential(
            *[blk(wf, **blk_params) for _ in range(mid_blk_nums)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(wf, wf * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                ))
            wf = wf // 2
            self.decoders.append(
                nn.Sequential(
                    *[blk(wf, **blk_params) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)


    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def __block__(self, blk_name):
        pass