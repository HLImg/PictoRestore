# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:23
# @Author : Liang Hao
# @FileName : __init__.py
# @Email : lianghao@whu.edu.cn

"""
The difference between the rewritten transforms and the orginal
ones lies in the processing of ndarray data instead of PIL images.
In addition, our transforms are designed to support the input of
single or multiple images.

The referenced code are as follows:
- [BSRGAN-RGB](https://github.com/cszn/BSRGAN/blob/main/utils/utils_image.py)
- [SERT-HSI](https://github.com/MyuLi/SERT)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
"""

from .basics import (Compose, ToImage, ToUint8, ToUint16, ToNdarray,
                     ToTensor, CenterCrop, RandomCrop, Identity,
                     Uint16ToSingle, Uint8ToSingle, ToNdarray_chw2hwc)

from .noise import (NIIDGaussianNoise, IIDGaussianNoise, PoissonNoise,
                    JPEGNoise, ImpulseNoise, StripeNoise, DeadLineNoise,
                    ComplexNoise)

from .augment import (Flip, FlipUD, FlipLR, RandomFlip, RandomRotation)

from .downsample import (Resize, RandomResize)

__all__ = [
    'Compose', 'ToImage', 'ToUint8', 'ToUint16', 'ToNdarray', 'ToTensor',
    'CenterCrop', 'RandomCrop', 'NIIDGaussianNoise', 'IIDGaussianNoise',
    'PoissonNoise', 'JPEGNoise', 'ImpulseNoise', 'StripeNoise', 'DeadLineNoise',
    'ComplexNoise', 'Flip', 'FlipUD', 'FlipLR', 'RandomFlip', 'RandomRotation',
    'Identity', 'Uint16ToSingle', 'Uint8ToSingle', 'ToNdarray_chw2hwc'
]
