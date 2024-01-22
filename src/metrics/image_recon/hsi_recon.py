# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:00
# @Author : Liang Hao
# @FileName : hsi_recon.py
# @Email : lianghao@whu.edu.cn

import torch
import imgvision as iv

from src.utils import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class PSNR_HSI:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border

    def __call__(self, input, target, **kwargs):
        assert input.shape == target.shape, (f"the shapes of input "
                                             f"{input.shape} and target {target.shape} are different.")

        if len(input.shape) != 3:
            raise (f"In the metric for hyperspectral or remote image, the shape of input must be "
                   f"(h, w, c), but received {input.shape} and {target.shape}")

        if self.crop_border != 0:
            input = input[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]
            target = target[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]

        metric = iv.spectra_metric(input, target)
        return metric.PSNR(**kwargs)


@METRIC_REGISTRY.register()
class SSIM_HSI:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border

    def __call__(self, input, target, **kwargs):
        assert input.shape == target.shape, (f"the shapes of input "
                                             f"{input.shape} and target {target.shape} are different.")

        if len(input.shape) != 3:
            raise (f"In the metric for hyperspectral or remote image, the shape of input must be "
                   f"(h, w, c), but received {input.shape} and {target.shape}")

        if self.crop_border != 0:
            input = input[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]
            target = target[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]

        metric = iv.spectra_metric(input, target)
        return metric.SSIM(**kwargs)


@METRIC_REGISTRY.register()
class SAM:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border

    def __call__(self, input, target, **kwargs):
        assert input.shape == target.shape, (f"the shapes of input "
                                             f"{input.shape} and target {target.shape} are different.")

        if len(input.shape) != 3:
            raise (f"In the metric for hyperspectral or remote image, the shape of input must be "
                   f"(h, w, c), but received {input.shape} and {target.shape}")

        if self.crop_border != 0:
            input = input[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]
            target = target[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]

        metric = iv.spectra_metric(input, target)
        return metric.SAM(**kwargs)


@METRIC_REGISTRY.register()
class ERGAS:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border

    def __call__(self, input, target, **kwargs):
        assert input.shape == target.shape, (f"the shapes of input "
                                             f"{input.shape} and target {target.shape} are different.")

        if len(input.shape) != 3:
            raise (f"In the metric for hyperspectral or remote image, the shape of input must be "
                   f"(h, w, c), but received {input.shape} and {target.shape}")

        if self.crop_border != 0:
            input = input[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]
            target = target[self.crop_border: -self.crop_border, self.crop_border: -self.crop_border, :]

        metric = iv.spectra_metric(input, target)
        return metric.ERGAS(**kwargs)