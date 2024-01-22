# -*- coding: utf-8 -*-
# @Time    : 12/25/23 9:50 PM
# @File    : downsample.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import random

import cv2 as cv
import numpy as np

from .basics import BasicObject


class Resize(BasicObject):
    def __init__(self, scale=1, inter_mode='cubic', down=False):
        if inter_mode.lower() == 'cubic':
            self.mode = cv.INTER_CUBIC
        elif inter_mode.lower() == 'linear':
            self.mode = cv.INTER_LINEAR
        elif inter_mode.lower() == 'nearst':
            self.mode = cv.INTER_NEAREST
        else:
            raise ValueError(f"Unknown interpolation mode: {inter_mode}")

        self.scale = scale
        if down:
            self.scale = 1 / scale

    def __call__(self, *images):
        res = []
        for image in images:
            h = int(image.shape[0] * self.scale)
            w = int(image.shape[1] * self.scale)
            res.append(cv.resize(image, dsize=(w, h), interpolation=self.mode))

        return self.return_list(res)


class RandomResize(BasicObject):
    def __init__(self, scale=4, down_p=0.7, up_p=0.8, clip=True):
        self.scale = scale
        self.down_p = down_p
        self.up_p = up_p
        self.clip = clip

    def __call__(self, *images):
        rnum = np.random.rand()
        if rnum > self.up_p:
            scale = random.uniform(1, 2)
        elif rnum < self.down_p:
            scale = random.uniform(0.5 / self.scale, 1)
        else:
            scale = 1.

        res = []
        for image in images:
            h = int(image.shape[0] * scale)
            w = int(image.shape[1] * scale)
            image = cv.resize(image, dsize=(w, h), interpolation=random.choice([
                                     cv.INTER_CUBIC, cv.INTER_LINEAR, cv.INTER_NEAREST
                                 ])
            )
            if self.clip:
                image = np.clip(image, 0.0, 1.0)
            res.append(image)

        return self.return_list(res)