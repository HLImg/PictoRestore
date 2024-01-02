# -*- coding: utf-8 -*-
# @Time    : 01/02/24 8:34 PM
# @File    : synthetic_noise_rgb.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import numpy as np
import src.datasets.transforms as transforms

from src.datasets import SingleDataset
from src.utils import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SynRGBNoise(SingleDataset):
    def __init__(self, 
                 hq_path, 
                 down_scale=1, 
                 patch_size=-1, 
                 aug_mode=None, 
                 read_mode='disk', 
                 data_type='uint8',
                 noise_clip= True,
                 noise_name='Gaussian',
                 noise_range=(5, 60, 5),
                 noise_seed=2018):
        super().__init__(hq_path, down_scale, patch_size, aug_mode, read_mode, data_type)
        
        if noise_name.lower() == 'poisson':
            self.add_noise = transforms.PoissonNoise(
                clip=noise_clip, noise_range=noise_range, seed=noise_seed
            )
        elif noise_name.lower() == 'gaussian':
            self.add_noise = transforms.IIDGaussianNoise(
                clip=noise_clip, noise_range=noise_range, seed=noise_seed
            )
        else:
            raise ValueError(f"Only support Poisson and Gaussian, but received {noise_name}")
        
        self.transform_hq = transforms.Compose(
            [
                transforms.Uint8ToSingle(),
                self.augment
            ]
        )
            
        
        self.transform_lq_hq = transforms.Compose(
            [
                
                transforms.RandomCrop(size=patch_size, down_scale=down_scale),
                transforms.ToTensor()
            ]
        )
        
    
    def __getitem__(self, item):
        self.hq_params['item'] = item
        img_hq = self.get_image(self.hq_params)
        img_hq = self.transform_hq(img_hq)
        img_lq = self.add_noise(img_hq)
        img_hq = img_hq.astype(np.float32)
        img_lq = img_lq.astype(np.float32)
        img_lq, img_hq = self.transform_lq_hq(img_lq, img_hq)
        return {'lq': img_lq, 'hq': img_hq}
        
        
        