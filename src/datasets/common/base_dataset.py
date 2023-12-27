# -*- coding: utf-8 -*-
# @Time : 2023/12/27
# @Author : Liang Hao
# @FileName : base_dataset
# @Email : lianghao@whu.edu.cn
import os

import cv2
import lmdb
import glob
import random
import numpy as np
import os.path as osp

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
    def get_lmdb_info(self, lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = osp.join(lmdb_path, 'meta_info.txt')
        assert osp.exists(path), f"the lmdb path named {path} is not exist"
        info = []
        with open(path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split(' ')
                line[0] = line[0].encode()
                line[1] = eval(line[1])
                info.append(line)
        return env, info

    def get_disk_info(self, root_dir):
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

    def get_disk_img(self, file_paths, item):
        img_np = np.array(Image.open(file_paths[item]))
        return img_np

    def get_lmdb_img(self, env, key, shape, dtype=np.float32):
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        img_np = np.frombuffer(buf, dtype=dtype)
        img_np = img_np.reshape(shape)
        return img_np
