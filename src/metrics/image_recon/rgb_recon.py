# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:00
# @Author : Liang Hao
# @FileName : rgb_recon.py
# @Email : lianghao@whu.edu.cn

import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from src.utils import calculate_psnr, calculate_ssim, METRIC_REGISTRY

@METRIC_REGISTRY.register()
class PSNR_RGB:
    def __init__(self, crop_border=0, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_chanel = test_y_channel

    def __call__(self, input, target):
        assert input.dtype == target.dtype, f"the dtypes of input and target are different."
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray), f"input and target must be np.nadarray, and shape is (h, w, c)"
            
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
        assert input.dtype == target.dtype, f"the dtypes of input and target are different."
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray), f"input and target must be np.nadarray, and shape is (h, w, c)"
        
        
        return calculate_ssim(img1=input,
                              img2=target,
                              crop_border=self.crop_border,
                              input_order=self.input_order,
                              test_y_channel=self.test_y_chanel)