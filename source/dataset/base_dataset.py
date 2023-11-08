# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 9:47
# @File    :   base_dataset.py
# @Email   :   lianghao@whu.edu.cn

import cv2
import lmdb
import glob
import random
import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

class BaseDataSet(Dataset):
    def __init__(self, scale=1, patch_size=-1):
        """
        Notes:
            self.patch_size表示低分辨率图像的patch
            高分辨率图像的patch大小为 self.patch_size * self.scale
        Args:
            scale: 表示超分数据集还是同等分辨率的数据，scale=1表示退化前后，图像分辨率没有变换
            patch_size: 低分辨率图像块的大小，当patch_size=-1时，用来表示当前是验证集，不需要切patch
        """
        self.scale = scale
        self.patch_size = patch_size

    def get_lmdb_info(self, lmdb_path):
        """获取lmdb文件的信息
        Args:
            lmdb_path: lmdb文件路径
        Returns:
            env : lmdb文件的环境
            keys: lmdb文件的索引值
        """
        env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
        path = osp.join(lmdb_path, 'meta_info.txt')
        assert osp.exists(path), f"the lmdb file named {path} is not exits"
        with open(path, 'r') as file:
            keys = [line.strip().split(".")[0].encode() for line in file.readlines()]
        return env, keys

    def get_lmdb_img(self, keys, index, env):
        """获取lmdb中的图像
        Args:
            env: lmdb的环境
            keys: lmdb的索引值
            index: 指定的序号
        Returns:
            img_np : 指定的图像，uint8
        """
        with env.begin(write=False) as txn:
            buf = txn.get(keys[index])
        img_np = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img_np

    def get_disk_info(self, disk_dir):
        """获取磁盘上某个路径下的所有图像路径
        disk_dir下仅包含图像文件
        Args:
            disk_dir: 路径
        Returns:
            keys: 所有图像的路径
        """
        assert osp.exists(disk_dir), f"{disk_dir} is not exits"
        keys = [path for path in glob.glob(osp.join(disk_dir, '*'))]
        keys = np.array(sorted(keys))
        return keys

    def get_disk_img(self, keys, index):
        """读取磁盘上的图像
        Args:
            keys: 所有图像的路径
            index: 指定图像的序号
        Returns:
            img: 指定的图像
        """
        img = Image.open(keys[index])
        img_np = np.array(img)
        return img_np

    def get_img(self, keys, index, env=None):
        """读取指定index的图像
        Args:
            keys: 图像的路径或者索引值
            index: 指定图像序号
            env: lmdb的环境，如果是disk读取的话，不用指定
        Returns:
            img: 指定的图像
        """
        if not env:
            # 读取磁盘上的图像
            img = self.get_disk_img(keys, index)
        else:
            # 读取lmdb中的图像
            img = self.get_lmdb_img(keys, index, env=env)

        return img

    def random_img2patch(self, img_lq, img_hq):
        """随机切patch
        可用于超分数据集, 通过self.scale控制缩放
        Args:
            img_lq: 低质量（分辨率）图像
            img_hq: 高质量（分辨率）图像

        Returns:
            patch_lq : 低质量（分辨率）图像块
            patch_hq : 高质量（分辨率）图像块
        """
        lh, lw = img_lq.shape[:2]
        hh, hw = img_hq.shape[:2]

        ind_h = random.randint(0, max(min(lh, hh // self.scale), self.patch_size) - self.patch_size)
        ind_w = random.randint(0, max(min(lw, hw // self.scale), self.patch_size) - self.patch_size)

        patch_lq = img_lq[ind_h:ind_h + self.patch_size, ind_w: ind_w + self.patch_size]
        patch_hq = img_hq[ind_h * self.scale:(ind_h + self.patch_size) * self.scale,
                          ind_w * self.scale:(ind_w + self.patch_size) * self.scale]
        return patch_lq, patch_hq

    def random_augment(self, *imgs, flip=True, rot=True):
        """随机增强图像数据
        Args:
            *imgs: 支持多个图像
            flip: 选择是否flip
            rot: 选择是否rotation
        Returns:
            augmented: 进行相同增强后的图像

        """
        def _flip(img, mode=0):
            if mode == 0:
                return img
            if mode == 1:
                return np.flip(img)
            if mode == 2:
                return np.fliplr(img)
            if mode == 3:
                return np.flipud(img)

        def _rotation(img, k=0):
            return np.rot90(img, k)

        if rot:
            rot_k = random.randint(0, 3)

        if flip:
            flip_mode = random.randint(0, 3)

        augmented = []

        for img in imgs:
            if flip:
                img = _flip(img, flip_mode)
            if rot:
                img = _rotation(img, rot_k)
            augmented.append(img)
        return augmented










