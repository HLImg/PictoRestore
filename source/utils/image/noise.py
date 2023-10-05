# -*- coding: utf-8 -*-
# @Time    : 9/13/23 7:44 PM
# @File    : noise.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import numpy as np

class AddMathNoise:
    def __init__(self, noise_type='gaussian', clip=False):
        self.clip = clip
        if noise_type.lower() == 'gaussian':
            self.add_noise = self.gaussian_noise
        elif noise_type.lower() == 'poisson':
            self.add_noise = self.poisson_noise
        elif noise_type.lower() == 'niid-gaussian':
            self.add_noise = self.nid_gaussian_noise
        else:
            raise f'noise type {noise_type} is not exits in math-noise'

    def gaussian_noise(self, img, noise_level: object):
        noise = np.random.randn(*img.shape) * (noise_level / 255.)
        return noise

    def poisson_noise(self, img, noise_level: object):
        noise = img - (np.random.poisson(img * noise_level) / noise_level).astype(img.dtype)
        return noise

    def nid_gaussian_noise(self, img, noise_level: object):
        sigmas = np.array(noise_level) / 255.
        bwsigma = np.reshape(sigmas[np.random.randint(0, len(sigmas), img.shape[0])], (-1, 1, 1))
        noise = np.random.randn(*img.shape) * bwsigma
        return noise

    def __call__(self, img, noise_level):

        noised = img + self.add_noise(img, noise_level)

        if self.clip:
            noised = noised.clip(0, 1)

        return noised

class AddBlindGaussianNoise(object):
    def __init__(self, min_sigma, max_sigma, clip=False):
        self.clip = clip
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img, noise_level=None):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255.
        noise = np.random.randn(*img.shape) * sigma
        noised = img + noise
        if self.clip:
            noised = noised.clip(0, 1)
        return noised

class AddMixedNoise(object):
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.num_bands = num_bands
        self.noise_bank = noise_bank
    def __call__(self, img):
        pos = 0
        b, h, w = img.shape
        all_bands = np.random.permutation(range(b))
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * b))
            bands = all_bands[pos: pos + num_band]
            pos = pos + num_band
            img = noise_maker(img, bands)

        return img

class DeadLineNoise(object):
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount

        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        b, h, w = img.shape
        num_deadline = np.random.randint(np.ceil(self.min_amount * w),
                                         np.ceil(self.max_amount * w),
                                         len(bands))

        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(w))
            loc = loc[:n]
            img[i, :, loc] = 0

        return img

class ImpluseNoise(object):
    def __init__(self, amount, s_vs_p=0.5):
        self.s_vs_p = s_vs_p
        self.amount = np.array(amount)

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out

    def __call__(self, img, bands):
        bwamounts = self.amount[np.random.randint(0, len(self.amount), len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i,...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img


class StripeNoise(object):
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        b, h, w = img.shape
        num_stripe = np.random.randint(np.floor(self.min_amount * w),
                                       np.floor(self.max_amount * w),
                                       len(bands))

        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(w))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img
