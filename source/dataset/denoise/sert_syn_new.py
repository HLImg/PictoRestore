# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/20 21:02:18
# @File    :   sert_syn_new.py
# @Contact   :   lianghao@whu.edu.cn

import re
from typing import Any
import lmdb
import threading
import numpy as np
import os.path as osp

from source.utils.image.add_noise import *
from torchvision.transforms import Compose
from source.utils.image.transpose import img2tensor
from source.dataset.base_dataset import BaseDataSet

class SERTDataSetMix(BaseDataSet):
    def __init__(self, hq_path, lq_path=None ,scale=1, patch_size=-1,
                 flip=False, rotation=False, noise_type=['nidd'],noise_level=[10, 30, 50, 70],
                 noise_clip=False, norm=False):
        super(SERTDataSetMix, self).__init__(scale, patch_size)
        
        self.flip = flip
        self.norm = norm
        self.rotation = rotation
        self.noise_clip = noise_clip
        self.noise_level = noise_level
        self.match_shape = re.compile(r"\((\d+),\s*(\d+),\s*(\d+)\)")
        
        self.is_test = False        
        self.hq_env, self.hq_keys, self.hq_shapes = self.get_lmdb_info(hq_path)
        if lq_path:
            self.is_test = True
            self.lq_env, self.lq_keys, self.lq_shapes = self.get_lmdb_info(lq_path)
        
        add_noise = {
            'niid': lambda x: x,
            'stripe': AddStripeNoise(),
            'impluse': AddImpluseNoise(),
            'deadline': AddDeadLineNoise()
        }
        
        if not self.is_test:
            transforms = []
            for name in noise_type:
                transforms.append(add_noise[name])
                
            self.add_noise = self.add_noise = Compose([
            AddNoiseNoniid(sigmas=self.noise_level, clip=self.noise_clip),
            SequentialSelect(
                transforms=transforms
            )
        ])
    
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
        if self.norm:
            img_np = self.norm_max_min(img_np)
        return img_np
    
    def get_lmdb_img(self, item):
        hq_np = self.get_lmdb_img_(self.hq_env, self.hq_keys[item], self.hq_shapes[item])
        hq_np = np.transpose(hq_np, (1, 2, 0))
        
        if self.is_test:
            lq_np = self.get_lmdb_img_(self.lq_env, self.lq_keys[item], self.lq_shapes[item])
            lq_np = np.transpose(lq_np, (1, 2, 0))
        else:
            lq_np = self.add_noise(hq_np.copy())
        
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
    

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()

class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n
            
    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))
    
    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out

class AddNoiseNoniid(object):
    def __init__(self, sigmas, clip=False) -> None:
        self.sigmas = np.array(sigmas) / 255.
        self.clip = clip
    
    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[2])], (1,1, -1))
        noise = np.random.randn(*img.shape) * bwsigmas
        noised = img + noise
        
        if self.clip:
            noised = noised.clip(0, 1)
            
        return noised
        