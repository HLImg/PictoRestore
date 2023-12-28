# -*- coding: utf-8 -*-
# @Time : 2023/12/26
# @Author : Liang Hao
# @FileName : augment
# @Email : lianghao@whu.edu.cn

import random
import numpy as np

from .basics import BasicObject, Compose


class Flip(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            res.append(np.flip(image))

        return self.return_list(res)


class FlipLR(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            res.append(np.fliplr(image))

        return self.return_list(res)


class FlipUD(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            res.append(np.flipud(image))

        return self.return_list(res)


class RandomRotation(BasicObject):
    def __call__(self, *images):
        k = random.randint(0, 3)
        res = []
        for image in images:
            res.append(np.rot90(image, k=k))

        return self.return_list(res)


class RandomFlip(BasicObject):
    def __init__(self, p=0):
        self.transform = Compose(
            [
                lambda *images: images,
                Flip(),
                FlipLR(),
                FlipUD()
            ], p=p, k=1
        )

    def __call__(self, *images):
        return self.transform(images)