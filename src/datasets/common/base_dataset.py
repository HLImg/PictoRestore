# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:25
# @Author : Liang Hao
# @FileName : base_dataset.py
# @Email : lianghao@whu.edu.cn


import os
import cv2
import lmdb
import glob
import random
import numpy as np
import os.path as osp
import src.datasets.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class BaseDataSet(Dataset):
    """
    There are three different modes for loading training or testing data.
    (1) LMDB:
        ├── data.mdb
        ├── lock.mdb
        └── meta_info.txt
            ├── key1 (512, 512, 3)
            ├── key2 (512, 512, 3)
        :returns (env, info=[[key1, shape, ...], [key2, shape, ...], ])
    (2) DISK of image files:
        test_96
        ├── bulb_0822-0909.mat
        ├── CC_40D_2_1103-0917.mat
        ├── gavyam_0823-0930.mat
        :returns (data=[(bulb_0822-0909.mat, None)], 0)
        or
        test_gaussian
        ├── 512_10
        │   ├── bulb_0822-0909.mat
        │   ├── CC_40D_2_1103-0917.mat
        ├── 512_30
        │   ├── bulb_0822-0909.mat
        │   ├── CC_40D_2_1103-0917.mat
        :returns (data=[(bulb_0822-0909.mat, "512_10"), ], num_class)
    """

    def __init__(self,
                 data_type="uint8",
                 read_mode='disk',
                 aug_mode=['flip', 'rot']):
        if aug_mode == None:
            self.augment = transforms.Identity()

        if isinstance(aug_mode, (tuple, list)):
            if len(aug_mode) == 0:
                self.augment = transforms.Identity()
            elif len(aug_mode) == 1:
                aug_mode = aug_mode[0]
            elif len(aug_mode) == 2:
                self.augment = transforms.Compose(
                    [
                        transforms.RandomRotation(),
                        transforms.RandomFlip(p=-1)
                    ], p=0.5, k=1
                )
            else:
                raise ValueError(f"Only support 'flip' and 'rot', but received {aug_mode}")

        if isinstance(aug_mode, str):
            if aug_mode.lower() == 'rot':
                self.augment = transforms.RandomRotation()
            elif aug_mode.lower() == 'flip':
                self.augment = transforms.RandomFlip(p=-1)
            else:
                raise ValueError(f"Unknown augmentation {aug_mode}, and only support 'flip' and 'rot'")

        self.read_mode = read_mode
        self.data_type = data_type

        if self.data_type == "uint8":
            self.trans2float32 = transforms.Uint8ToSingle()
        elif self.data_type == "uint16":
            self.trans2float32 = transforms.Uint16ToSingle()
        elif self.data_type == 'float32':
            self.trans2float32 = transforms.Identity()
        elif self.data_type == 'float64':
           self.trans2float32 = transforms.Identity()
        else:
            raise ValueError(f"Unknown data type {data_type}, only support uint8, uint16, float32, float64")



    def get_lmdb_info(self, lmdb_path):
        """
        Load lmdb to memory, including envs and infos
        LMDB:
        ├── data.mdb
        ├── lock.mdb
        └── meta_info.txt
            ├── key1 (512, 512, 3)
            ├── key2 (512, 512, 3)
        :param lmdb_path: path of lmdb file
        :returns (env, info=[[key1, shape, ...], [key2, shape, ...], ])
        """
        env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = osp.join(lmdb_path, 'meta_info.txt')
        assert osp.exists(path), f"the lmdb path named {path} is not exist"
        info = []
        with open(path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split('-')
                info.append({'key': line[0].encode(),
                             'shape': eval(line[1])})
        return env, info

    def get_disk_info(self, root_dir):
        """
        test_gaussian
        ├── 512_10
        │   ├── bulb_0822-0909.mat
        │   ├── CC_40D_2_1103-0917.mat
        ├── 512_30
        │   ├── bulb_0822-0909.mat
        │   ├── CC_40D_2_1103-0917.mat
        :returns (data=[(bulb_0822-0909.mat, "512_10"), ], num_class)
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", '.mat'}
        data = []
        num_class = 0
        for entry in os.scandir(root_dir):
            if entry.is_file() and any(entry.name.lower().endswith(ext)
                                       for ext in image_extensions):
                data.append((entry.path, None))
            elif entry.is_dir():
                sub_dir = entry.path
                class_name = entry.name
                num_class = num_class + 1
                for file_name in os.listdir(sub_dir):
                    if any(file_name.lower().endswith(ext) for ext in image_extensions):
                        full_path = osp.join(sub_dir, file_name)
                        data.append((full_path, class_name))

        return data, num_class

    def get_disk_img(self, paths, item):
        img_np = np.array(Image.open(paths[item][0]))
        return img_np

    def get_lmdb_img(self, envs, infos, item):
        with envs.begin(write=False) as txn:
            buf = txn.get(infos[item]['key'])
        img_np = np.frombuffer(buf, dtype=self.data_type)
        img_np = img_np.reshape(infos[item]['shape'])
        return img_np

    def get_image(self, params):
        if self.read_mode.lower() == 'lmdb':
            image_np = self.get_lmdb_img(**params)
        elif self.read_mode.lower() == 'disk':
            image_np = self.get_disk_img(**params)
        else:
            raise ValueError(f"Unknown read mode: {self.read_mode}, Only support lmdb and disk")
        return image_np
