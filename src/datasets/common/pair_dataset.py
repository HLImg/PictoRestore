# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : pair_dataset
# @Email : lianghao@whu.edu.cn

import numpy as np
import src.datasets.transforms as transforms

from .base_dataset import BaseDataSet

from src.utils import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class PairDataset(BaseDataSet):
    def __init__(self,
                 lq_path,
                 hq_path,
                 down_scale=1,
                 patch_size=-1,
                 aug_mode=None,
                 read_mode='disk',
                 data_type="uint8"):
        """
        Please note that the operators for augmentation and normalization (which convert input data to np.float32)
        have been predefined.
        :param lq_path: the path of low-quality dataset
        :param hq_path: the path of high-quality dataset
        :param down_scale: the loading of individual data samples is forbidden
                                when the down_scale factor is not equal to 1.
        :param patch_size: if patch size is equal to 1, input images will not be cropped into patches.
        :param aug_mode: ['flip', 'rot'] | None | 'flip' | 'rot'
        :param read_mode: 'disk' | 'lmdb'
        :param data_type: "uint8" | "uint16"
        """
        super().__init__(aug_mode=aug_mode,
                         read_mode=read_mode,
                         data_type=data_type)

        # loading data
        self.lq_params, self.hq_params = {}, {}
        if read_mode.lower() == 'disk':
            self.lq_params['paths'], _ = self.get_disk_info(lq_path)
            self.hq_params['paths'], _ = self.get_disk_info(hq_path)
            self.length = len(self.hq_params['paths'])
        else:
            self.lq_params['envs'], \
                self.lq_params['infos'] = self.get_lmdb_info(lq_path)

            self.hq_params['envs'], \
                self.hq_params['infos'] = self.get_lmdb_info(hq_path)

            self.length = len(self.hq_params['infos'])

        # define the transformation about data
        self.transform = transforms.Compose([
            self.trans2float32,
            transforms.RandomCrop(size=patch_size, down_scale=down_scale),
            self.augment,
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        self.hq_params['item'] = item
        self.lq_params['item'] = item

        img_hq = self.get_image(self.hq_params)
        img_lq = self.get_image(self.lq_params)

        tensor_lq, tensor_hq = self.transform(img_lq, img_hq)

        return {'hq': tensor_hq, 'lq': tensor_lq}
