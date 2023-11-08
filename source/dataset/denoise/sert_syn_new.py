# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/20 21:02:18
# @File    :   sert_syn_new.py
# @Contact   :   lianghao@whu.edu.cn

import re
import lmdb
import torch
import threading
import numpy as np
import os.path as osp

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
            'impluse': AddNoiseImpulse(),
            'stripe': AddNoiseStripe(),
            'deadline': AddNoiseDeadline()
        }
        
        if not self.is_test:
            transforms = []
            for name in noise_type:
                transforms.append(add_noise[name])
                
            self.add_noise = Compose([
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

        if self.is_test:
            lq_np = self.get_lmdb_img_(self.lq_env, self.lq_keys[item], self.lq_shapes[item])
        else:
            lq_np = self.add_noise(hq_np.copy())
        
        return lq_np, hq_np
    
    def __len__(self):
        return len(self.hq_keys)
    
    def chw2hwc(self, x):
        return np.transpose(x, (1, 2, 0))
    
    def hwc2chw(self, x):
        return np.transpose(x, (2, 0, 1))
    
    def __getitem__(self, item):
        # c h w
        img_lq, img_hq = self.get_lmdb_img(item)
        img_lq, img_hq = self.chw2hwc(img_lq), self.chw2hwc(img_hq)
        
        if self.patch_size > 0:
            img_lq, img_hq = self.random_img2patch(img_lq, 
                                                   img_hq)
        
        if self.flip or self.rotation:
            img_lq, img_hq = self.random_augment(img_lq, img_hq, flip=self.flip, rot=self.rotation)
        
        img_lq, img_hq = self.hwc2chw(img_lq), self.hwc2chw(img_hq)
        
        tensor_lq = torch.from_numpy(np.float32(img_lq.copy()))
        tensor_hq = torch.from_numpy(np.float32(img_hq.copy()))
        
        
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

class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos+num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img

class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas, clip=False):
        self.clip = clip
        self.sigmas = np.array(sigmas) / 255.
    
    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])], (-1,1,1))
        noise = np.random.randn(*img.shape) * bwsigmas
        noised = img + noise
        if self.clip:
            noised = noised.clip(0, 1)
        return noised

class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):        
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1/3]


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.15)]
        self.num_bands = [1/3]

class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.15)]
        self.num_bands = [1/3]


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""
    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands,bwamounts):
            self.add_noise(img[i,...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

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


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount        
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount*W), np.floor(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*0.5-0.25            
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount        
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount*W), np.ceil(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img
