# -*- coding: utf-8 -*-
# @Time    : 9/13/23 8:30 PM
# @File    : add_noise.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

from .noise import *
from torchvision.transforms import Compose

class AddDeadLineNoise(AddMixedNoise):
    def __init__(self):
        self.num_bands = [1 / 3]
        self.noise_bank = [DeadLineNoise(0.05, 0.15)]

class AddImpluseNoise(AddMixedNoise):
    def __init__(self):
        self.num_bands = [1 / 3]
        self.noise_bank = [ImpluseNoise([0.1, 0.3, 0.5, 0.7])]

class AddStripeNoise(AddMixedNoise):
    def __init__(self):
        self.num_bands = [1 / 3]
        self.noise_bank = [StripeNoise(0.05, 0.15)]


class AddComplexNoise(AddMixedNoise):
    def __init__(self):
        self.num_bands = [1/ 3, 1 / 3, 1 / 3]
        self.noise_bank= [
            StripeNoise(0.05, 0.15),
            DeadLineNoise(0.05, 0.15),
            ImpluseNoise([0.1, 0.3, 0.5, 0.7])
        ]


class AddNoise2HSI:
    def __init__(self, noise_type, clip=False):
        self.type = noise_type
        
        if noise_type.lower() == 'deadline':
            self.add_noise = AddDeadLineNoise()

        elif noise_type.lower() == 'stripe':
            self.add_noise = AddStripeNoise()

        elif noise_type.lower() == 'complex':
            self.add_noise = AddComplexNoise()

        elif noise_type.lower() == 'mixture':
            self.add_nidd = AddMathNoise('niid-gaussian', clip=clip)
            self.add_complex = AddComplexNoise()

        elif noise_type.lower() == 'impulse':
            self.add_noise = AddImpluseNoise()

        elif noise_type.lower() == 'niid-gaussian':
            self.add_noise = AddMathNoise('niid-gaussian', clip=clip)

        elif noise_type.lower() == 'gaussian':
            self.add_noise = AddMathNoise('gaussian', clip=clip)

        elif noise_type.lower() == 'blind-gaussian':
            self.add_noise = AddBlindGaussianNoise(min_sigma=10, max_sigma=70, clip=clip)

        else:
            raise f"the noise type named { noise_type} is not exits in properties"

    def __call__(self, img, noise_level=10):
        if self.type.lower() in ['gaussian', 'niid-gaussian', 'blind-gaussian']:
            return self.add_noise(img, noise_level)
        elif self.type.lower() == 'mixture':
            return self.add_complex(self.add_nidd(img, noise_level))
        else:
            return self.add_noise(img)