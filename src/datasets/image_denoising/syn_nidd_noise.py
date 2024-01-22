# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:35
# @Author : Liang Hao
# @FileName : syn_nidd_noise.py
# @Email : lianghao@whu.edu.cn

import os
from typing import Any
import cv2
import glob
import lmdb
import torch
import pickle
import random
import numpy as np

from PIL import Image
from skimage import img_as_float32
from torch.utils.data import Dataset
from src.utils import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SimuteNIDDGaussian(Dataset):
    def __init__(self,
                 hq_path,
                 length,
                 patch_size=128,
                 augment=False,
                 sigma_min=0,
                 sigma_max=75,
                 sigma_iid=[30, 50, 70]):
        super().__init__()

        self.augment = augment
        self.pch_size = patch_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_idd = sigma_iid

        self.im_list = [path for path in glob.glob(os.path.join(hq_path, '*'))]

        self.length = length

        self.num_images = len(self.im_list)

    def __len__(self):
        return self.length

    def gaussian_kernel(self, H, W, center, scale):
        centerH = center[0]
        centerW = center[1]
        XX, YY = np.meshgrid(np.arange(W), np.arange(H))
        ZZ = 1 / (2 * np.pi * scale ** 2) * np.exp((-(XX - centerH) ** 2 - (YY - centerW) ** 2) / (2 * scale ** 2))
        return ZZ

    def generate_nidd_sigma(self):
        pch_size = self.pch_size
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size / 4, pch_size / 4 * 3)
        kernel = self.gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        down = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        if up < down:
            up, down = down, up
        up += 5 / 255.0
        sigma_map = down + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (up - down)
        sigma_map = sigma_map.astype(np.float32)
        return sigma_map[:, :, np.newaxis]

    def generate_idd_sigma(self):
        sigma = random.choice(self.sigma_idd) / 255.
        sigma_map = np.ones(self.pch_size, self.pch_size) * sigma
        sigma_map = sigma_map.astype(np.float32)
        return sigma_map[:, :, np.newaxis]

    def generate_sigma(self):
        flag = random.randint(1, 2)

        if flag == 1 and not self.sigma_idd:
            return self.generate_idd_sigma()
        else:
            return self.generate_nidd_sigma()

    def crop_patch(self, im):
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H - self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        pch = im[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size]
        return pch

    def __getitem__(self, item):
        ind_im = random.randint(0, self.num_images - 1)
        im_ori = cv2.imread(self.im_list[ind_im], 1)[:, :, ::-1]
        im_gt = img_as_float32(self.crop_patch(im_ori))
        C = im_gt.shape[2]

        sigma_map = self.generate_sigma()
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        im_noisy = im_gt + noise.astype(np.float32)

        if self.augment:
            im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        sigma_map_gt = np.tile(sigma_map, (1, 1, C))
        sigma_map_gt = np.where(sigma_map_gt < 1e-5, 1e-5, sigma_map_gt)
        sigma_map_gt = torch.from_numpy(sigma_map_gt.transpose((2, 0, 1)))

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return {'lq': im_noisy, 'hq': im_gt, 'sigma': sigma_map_gt}


@DATASET_REGISTRY.register()
class VDNSimuTest(Dataset):
    def __init__(self, lmdb_path):
        super().__init__()

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = os.path.join(lmdb_path, 'meta_info.txt')

        self.info = []
        with open(path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split(' ')
                self.info.append({
                    'key': line[0].encode(),
                    'shape': eval(line[1])
                })

    def __len__(self):
        return len(self.info)

    def get_lmdb_img(self, envs, key):
        with envs.begin(write=False) as txn:
            serialized_data = txn.get(key)
            data = pickle.loads(serialized_data)
        return data

    def __getitem__(self, item):
        data = self.get_lmdb_img(self.env, key=self.info[item]['key'])

        im_gt = (data['im_gt']).astype(np.float32)
        im_noisy = (im_gt + data['noise']).astype(np.float32)

        sigma_map = (data['sigma'][:, :, np.newaxis]).astype(np.float32)
        sigma_map_gt = np.tile(sigma_map, (1, 1, 3))
        sigma_map_gt = np.where(sigma_map_gt < 1e-5, 1e-5, sigma_map_gt)
        sigma_map_gt = torch.from_numpy(sigma_map_gt.transpose((2, 0, 1)))

        im_gt = torch.from_numpy(np.transpose(im_gt, (2, 0, 1)))
        im_noisy = torch.from_numpy(np.transpose(im_noisy, (2, 0, 1)))

        return {'lq': im_noisy, 'hq': im_gt, 'sigma': sigma_map_gt}


def random_augmentation(*args):
    out = []
    if random.randint(0, 1) == 1:
        flag_aug = random.randint(1, 7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out


def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        pass
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out