# -*- coding: utf-8 -*-
# @Time    : 9/14/23 10:09 PM
# @File    : remote.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch
import numpy as np

def cal_sam_np(im1, im2, eps=1e-8):
    im1 = torch.squeeze(im1.data).cpu().numpy()
    im2 = torch.squeeze(im2.data).cpu().numpy()

    sam = (np.sum(im1 * im2, axis=0) + eps) / (np.sqrt(np.sum(im1 ** 2, axis=0)) + eps) / (np.sqrt(np.sum(im2 ** 2, axis=0)) + eps)

    return np.mean(np.real(np.arccos(sam)))

def cal_sam_torch(im1, im2, eps=1e-8):
    im1 = torch.squeeze(im1)
    im2 = torch.squeeze(im2)

    sam_1 = torch.sum(im1 * im2, dim=0) + eps
    sam_2 = torch.sqrt(torch.sum(im1 ** 2, dim=0)) + eps
    sam_3 = torch.sqrt(torch.sum(im2 ** 2, dim=0)) + eps

    sam = sam_1 / (sam_2 * sam_3)
    sam = torch.mean(torch.real(torch.arccos(sam)))

    return sam