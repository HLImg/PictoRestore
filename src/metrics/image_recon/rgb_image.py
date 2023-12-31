# -*- coding: utf-8 -*-
# @Time : 2023/12/31
# @Author : Liang Hao
# @FileName : rgb_image
# @Email : lianghao@whu.edu.cn

import cv2
import torch
import numpy as np

from src.utils import calculate_psnr, calculate_ssim, METRIC_REGISTRY

@METRIC_REGISTRY.register()
class PSNR_RGB:
    def __init__(self, crop_border=0, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_chanel = test_y_channel

    def __call__(self, input, target):
        return calculate_psnr(img1=input,
                              img2=target,
                              crop_border=self.crop_border,
                              input_order=self.input_order,
                              test_y_channel=self.test_y_chanel)


@METRIC_REGISTRY.register()
class SSIM_RGB:
    def __init__(self, crop_border=0, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_chanel = test_y_channel

    def __call__(self, input, target):
        return calculate_ssim(img1=input,
                              img2=target,
                              crop_border=self.crop_border,
                              input_order=self.input_order,
                              test_y_channel=self.test_y_chanel)