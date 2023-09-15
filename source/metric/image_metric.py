# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 14:00
# @File    :   image_metric.py
# @Email   :   lianghao@whu.edu.cn
# @Thanks  :   BasicSR

from .remote import *
from .psnr_ssim import *


class PSNR:
    def __init__(self, crop_border=0, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel

    def __call__(self, im1, im2):
        return calculate_psnr(im1, im2,
                              self.crop_border,
                              input_order=self.input_order,
                              test_y_channel=self.test_y_channel)


class SSIM:
    def __init__(self, crop_border=0, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel

    def __call__(self, im1, im2):
        return calculate_ssim(im1, im2,
                              self.crop_border,
                              input_order=self.input_order,
                              test_y_channel=self.test_y_channel)


class SAM:
    def __init__(self, crop_border=0, epsilon=1e-8):
        self.epsilon = epsilon
        self.crop_border = crop_border

    def __call__(self, im1, im2):
        return cal_sam_torch(im1, im2, eps=self.epsilon)