# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 14:00
# @File    :   image_metric.py
# @Email   :   lianghao@whu.edu.cn
# @Thanks  :   BasicSR

from .remote import *
from .psnr_ssim import *
from source.utils.image.transpose import reorder_image, to_y_channel

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
