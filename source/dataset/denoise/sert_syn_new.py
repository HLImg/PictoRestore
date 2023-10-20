# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/20 21:02:18
# @File    :   sert_syn_new.py
# @Contact   :   lianghao@whu.edu.cn

import re
import lmdb
import numpy as np
import os.path as osp

from source.utils.image.transpose import img2tensor
from source.dataset.base_dataset import BaseDataSet
from source.utils.image.add_noise import AddNoise2HSI

class SERTDataSetMix(BaseDataSet):
    def __init__(self, hq_path, lq_path=None ,scale=1, patch_size=-1,
                 flip=False, rotation=False, noise_level=[10, 30, 50, 70],
                 noise_clip=False, noise_type='gaussian'):
        super(SERTDataSetMix, self).__init__(scale, patch_size)
        
        self.flip = flip
        self.rotation = rotation
        self.noise_clip = noise_clip
        self.noise_level = noise_level
        self.match_shape = re.compile(r"\((\d+),\s*(\d+),\s*(\d+)\)")
        
        self.is_test = False        
        self.hq_env, self.hq_keys, self.hq_shapes = self.get_lmdb_info(hq_path)
        if lq_path:
            self.is_test = True
            self.lq_env, self.lq_keys, self.lq_shapes = self.get_lmdb_info(lq_path)
        
        if not self.is_test:
            self.add_noise = None
    
    def get_lmdb_info(self, lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = osp.join(lmdb_path, 'meta_info.txt')
        assert osp.exists(path), f'the lmdb file named {path} has not meta info'
        keys, shapes = [], []
        with open(path, 'r') as file:
            for line in file.readlines():
                key, shape = line.strip().split(' ')
                keys.append(key.encode())
                match = self.match_shape.search(shape)
                shape = tuple(map(int, match.groups()))
                shapes.append(shape[::-1])
        return env, keys, shapes
    
    def norm_max_min(self, img):
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min)
    
    def get_lmdb_img_(self, env, key, shape):
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        img_np = np.frombuffer(buf, dtype=np.float32)
        img_np = img_np.reshape(shape)
        img_np = self.norm_max_min(img_np)
        return img_np
    
    def get_lmdb_img(self, item):
        hq_np = self.get_lmdb_img(self.hq_env, self.hq_keys[item], self.hq_shapes[item])
        hq_np = np.transpose(hq_np, (1, 2, 0))
        
        if self.is_test:
            lq_np = self.get_lmdb_img(self.hq_env, self.hq_keys[item], self.hq_shapes[item])
            lq_np = np.transpose(lq_np, (1, 2, 0))
        else:
            lq_np = self.add_noise(hq_np.copy(), noise_level=self.noise_level)
        
        return lq_np, hq_np
    
    def __len__(self):
        return len(self.hq_keys)
    
    def __getitem__(self, item):
        img_lq, img_hq = self.get_lmdb_img(item)
        
        if self.patch_size > 0:
            img_lq, img_hq = self.random_img2patch(img_lq, img_hq)
        
        if self.flip or self.rotation:
            img_lq, img_hq = self.random_augment(img_lq, img_hq, flip=self.flip, rot=self.rotation)
        
        tensor_lq = img2tensor(np.float32(img_lq))
        tensor_hq = img2tensor(np.float32(img_hq))
        
        return {'hq': tensor_hq, 'lq': tensor_lq}