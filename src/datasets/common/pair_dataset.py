# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : pair_dataset
# @Email : lianghao@whu.edu.cn

import src.datasets.transforms as transforms

from .base_dataset import BaseDataSet


class PairDataset(BaseDataSet):
    def __init__(self, lq_path, hq_path, scale=1, patch_size=-1, is_flip=-1, is_rot=-1):
        super().__init__()

        self.flip_augment = transforms.RandomFlip(p=-1)
        self.rot_augment = transforms.RandomRotation()
