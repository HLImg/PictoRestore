# -*- coding: utf-8 -*-
# @Time    : 9/15/23 4:18 PM
# @File    : hsi_syn.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import re
import cv2
import lmdb
import numpy as np
import os.path as osp
from source.utils.image.transpose import img2tensor
from source.dataset.base_dataset import BaseDataSet
from source.utils.image.add_noise import AddNoise2HSI

class ICVLDataset(BaseDataSet):
    def __init__(self, hq_path, scale=1, patch_size=-1, flip=False,
                 rotation=False, noise_level=[15, 25, 50], noise_clip=False, noise_type='gaussian'):
        super(ICVLDataset, self).__init__(scale, patch_size)
        
        self.flip = flip
        self.rotation = rotation
        self.noise_clip = noise_clip
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.match_shape = re.compile(r"\((\d+),\s*(\d+),\s*(\d+)\)")

        self.hq_env, self.hq_keys, self.hq_shapes = self.get_lmdb_info(hq_path)

        self.add_noise = AddNoise2HSI(noise_type=noise_type, clip=self.noise_clip)

    
    def get_lmdb_info(self, lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = osp.join(lmdb_path, 'meta_info.txt')
        assert osp.exists(path), f'the lmdb file named {path} has not meta info'
        keys, shapes = [], []
        with open(path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split('.mat')
                key = (line[0] + '.mat').encode()
                keys.append(key)
                match = self.match_shape.search(line[1])
                shape = tuple(map(int, match.groups()))
                shapes.append(shape)

        return env, keys, shapes
        
    def __len__(self):
        return len(self.hq_keys)

    def norm_max_min(self, img):
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min)

    def get_lmdb_img_(self, env, key, shape):
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        img_np = np.frombuffer(buf, dtype=np.float64)
        img_np = img_np.reshape(shape)
        img_np = self.norm_max_min(img_np)
        return img_np

    def get_lmdb_img(self, item):
        hq_np = self.get_lmdb_img_(self.hq_env, self.hq_keys[item], self.hq_shapes[item])
        lq_np = self.add_noise(hq_np.copy(), noise_level=self.noise_level)
        return lq_np, hq_np

    def __getitem__(self, item):
        img_lq, img_hq = self.get_lmdb_img(item)
        if self.patch_size > 0:
            img_lq, img_hq = self.random_img2patch(img_lq, img_hq)

        if self.flip or self.rotation:
            img_lq, img_hq = self.random_augment(img_lq, img_hq,
                                                 flip=self.flip, rot=self.rotation)

        tensor_lq = img2tensor(np.float32(img_lq))
        tensor_hq = img2tensor(np.float32(img_hq))

        return {'hq': tensor_hq, 'lq': tensor_lq}



