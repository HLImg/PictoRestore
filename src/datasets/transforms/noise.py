# -*- coding: utf-8 -*-
# @Time    : 12/25/23 7:46 PM
# @File    : noise.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

from __future__ import annotations

import cv2
import cv2 as cv
import numpy as np

from .basics import BasicObject


class BasicNoise(BasicObject):
    def __init__(self, clip, noise_range: [list | tuple | int | float], seed=42):
        """
        the basic class for adding noise to images
        :param clip: determine if noisy image should be clipped to the range of [0, 1].
        :param noise_range: the range of noise should be specified as a list, tuple, integer, or float.
        :param seed: the seed that should be used to ensure reproducible experiments.
        """
        np.random.seed(seed=seed)
        self.clip = clip
        noise_range = eval(noise_range)
        if isinstance(noise_range, list):
            self.noise_range = noise_range
        elif isinstance(noise_range, tuple):
            if len(noise_range) == 2:
                self.noise_range = np.arange(
                    start=noise_range[0],
                    stop=noise_range[1],
                    step=0.5
                )
            elif len(noise_range) == 3:
                self.noise_range = np.arange(
                    start=noise_range[0],
                    stop=noise_range[1],
                    step=noise_range[2]
                )
            else:
                raise TypeError(f"Only tuples of the form (start, end) or (start, end, step) \
                                are supported, but received {noise_range}")
        elif isinstance(noise_range, (int, float)):
            self.noise_range = [noise_range]
        else:
            raise TypeError(f"Only tuple, list, int or float are acceptable values, but received {noise_range}")

    def random_noise_level(self) -> int | float:
        return np.random.choice(self.noise_range)


class IIDGaussianNoise(BasicNoise):
    def __init__(self, clip, noise_range, seed=42):
        super().__init__(clip, noise_range, seed)

    def __call__(self, *images):
        noise_level = self.random_noise_level()
        noise = np.random.normal(loc=0, scale=noise_level / 255., size=images[0].shape)
        res = []
        for image in images:
            image = image + noise
            if self.clip:
                image = image.clip(0, 1)
            res.append(image)

        return self.return_list(res)


class PoissonNoise(BasicNoise):
    def __init__(self, clip, noise_range, seed=42):
        super().__init__(clip, noise_range, seed)

    def __call__(self, *images):
        noise_level = self.random_noise_level()
        res = []
        for image in images:
            image = np.random.poisson(image * noise_level) / noise_level
            if self.clip:
                image = image.clip(0, 1)
            res.append(image)

        return self.return_list(res)


class NIIDGaussianNoise(BasicNoise):
    def __init__(self, clip, noise_range, seed=42):
        super().__init__(clip, noise_range, seed)

    def __call__(self, *images):
        sigmas = np.array(self.noise_range) / 255.
        bw_sigma = np.reshape(sigmas[np.random.randint(0, len(sigmas),
                                                       images[0].shape[0])], (-1, 1, 1))
        noise = np.random.randn(*images[0].shape) * bw_sigma
        res = []
        for image in images:
            image = image + noise
            if self.clip:
                image = image.clip(0, 1)
            res.append(image)

        return self.return_list(res)


class JPEGNoise(BasicObject):
    def __init__(self, min_factor=30, max_factor=95):
        """
        Add JPEG compression noise to RGB images
        :param min_factor: minimum quality factor., 30
        :param max_factor: maximum quality factor, 95
        """
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, *images):
        quality_factor = np.random.randint(self.min_factor,
                                           self.max_factor)
        res = []
        for image in images:
            image = cv.cvtColor(self.single2uint8(image), cv.COLOR_RGB2BGR)
            result, encode_img = cv.imencode('.jpg', image,
                                         [int(cv.IMWRITE_JPEG_QUALITY), quality_factor])
            image = cv.imdecode(encode_img, 1)
            image = cv.cvtColor(self.uint8_to_single(image), cv.COLOR_BGR2RGB)
            res.append(image)

        return self.return_list(res)


class _BandWiseMixedNoise(BasicObject):
    def __init__(self, noise_makers: [], num_bands: []):
        """
        The basic class for add band-wise noise, it supports multiple kinds of
        band-wise noise.
        :param noise_makers: this is a list including different band-wise noise generates
        :param num_bands:
        """
        self.num_bands = num_bands
        self.noise_makers = noise_makers

    def __call__(self, *images):
        pos = 0
        b, h, w = images[0]
        all_bands = np.random.permutation(range(b))
        num = len(images)
        res = [image.copy for image in images]
        for noise_maker, num_band in zip(self.noise_makers, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * b))
            bands = all_bands[pos: pos + num_band]
            pos = pos + num_band
            res = noise_maker(res, bands=bands)
        return self.return_list(res)


class _DeadLineNoise(BasicObject):
    def __init__(self, min_amount: float, max_amount: float):
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, *images, bands=None):
        b, h, w = images[0].shape
        num_deadlines = np.random.randint(
            np.ceil(self.min_amount * w),
            np.ceil(self.max_amount * w),
            len(bands)
        )

        num = len(images)
        res = [image.copy() for image in images]

        for i, n in zip(bands, num_deadlines):
            loc = np.random.permutation(range(w))
            loc = loc[:n]
            for j in range(num):
                res[j][i, :, loc] = 0
        return self.return_list(res)


class DeadLineNoise(_BandWiseMixedNoise):
    def __init__(self, min_amount: float, max_amount: float, num_band: float | int):
        """
        Add Deadline noise to hyperspectral images.
        :param min_amount: 0.05
        :param max_amount: 0.15
        :param num_bands: 1 / 3
        """
        num_bands = [num_band]
        noise_makers = [_DeadLineNoise(min_amount=min_amount, max_amount=max_amount)]
        super().__init__(num_bands=num_bands, noise_makers=noise_makers)


class _ImpulseNoise(BasicObject):
    def __init__(self, amount: list, s_vs_p=0.5):
        self.s_vs_p = s_vs_p
        self.amount = np.array(amount)

    def add_noise(self, *images, axis, amount, salt_vs_pepper):
        res = []
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False],
                                   size=images[0].shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False],
                                  size=images[0].shape,
                                  p=[q, 1 - q])
        peppered = ~salted

        for image in images:
            image[axis, ...][flipped & salted] = 1
            image[axis, ...][flipped & peppered] = 0
            res.append(image)

        return self.return_list(res)

    def __call__(self, *images, bands=None):
        res = [image.copy() for image in images]
        bw_amounts = self.amount[np.random.randint(0, len(self.amount), len(bands))]
        for i, amount in zip(bands, bw_amounts):
            res = self.add_noise(res, axis=i, amount=amount, salt_vs_pepper=self.s_vs_p)
        return self.return_list(res)


class ImpulseNoise(_BandWiseMixedNoise):
    def __init__(self, amount: list, s_vs_p=0.5, num_band=1 / 3):
        """
        Add impulse noise to hyperspectral images.
        :param amount: [0.1, 0.3, 0.5, 0.7]
        :param s_vs_p: 0.5
        :param num_bands: 1/3
        """
        num_bands = [num_band]
        noise_makers = [_ImpulseNoise(amount=amount, s_vs_p=s_vs_p)]
        super().__init__(num_bands=num_bands, noise_makers=noise_makers)


class _StripeNoise(BasicObject):
    def __init__(self, min_amount, max_amount):
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, *images, bands=None):
        b, h, w = images[0].shape
        num_stripe = np.random.randint(
            np.floor(self.min_amount * w),
            np.floor(self.max_amount * w),
            len(bands)
        )

        num = len(images)
        res = [image for image in images]

        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(w))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            for j in range(num):
                res[j][i, :, loc] -= np.reshape(stripe, (-1, 1))

        return self.return_list(res)


class StripeNoise(_BandWiseMixedNoise):
    def __init__(self, min_amount, max_amount, num_band=1 / 3):
        """
        Add stripe noise to hyperspectral images
        :param min_amount: 0.05
        :param max_amount: 0.15
        :param num_bands: 1/3
        """
        num_bands = [num_band]
        noise_makers = [_StripeNoise(min_amount=min_amount, max_amount=max_amount)]
        super().__init__(num_bands=num_bands, noise_makers=noise_makers)


class ComplexNoise(_BandWiseMixedNoise):
    def __init__(self, stripe_params, deadline_params, impulse_params, num_bands: list):
        """
            Add complex noise including stripe, deadline and impulse
            noise to hyperspectral images.
        :param stripe_params: {}
        :param deadline_params: {}
        :param impulse_params: {}
        :param num_bands: [1 / 3, 1 / 3, 1 / 3]
        """
        if num_bands is None:
            num_bands = [1 / 3, 1 / 3, 1 / 3]
        noise_makers = [
            _StripeNoise(**stripe_params),
            _DeadLineNoise(**deadline_params),
            _ImpulseNoise(**impulse_params)
        ]
        
        super().__init__(num_bands=num_bands, noise_makers=noise_makers)
