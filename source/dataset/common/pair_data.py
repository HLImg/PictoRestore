# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 11:37
# @File    :   pair_data.py
# @Email   :   lianghao@whu.edu.cn

import cv2
from skimage import img_as_float32
from source.utils.image.transpose import img2tensor
from source.dataset.base_dataset import BaseDataSet

class PairDataset(BaseDataSet):
    def __init__(self, lq_path, hq_path, patch_size=-1, scale=1, flip=False, rotation=False, read_mode="disk"):
        """
        the dataset class for pair-data (lq-dataset, hq-dataset)
        Args:
            lq_path: 低质量（分辨率）图像
            hq_path: 高质量（高分辨率）图像
            patch_size: 低质量（分辨率）的图像块大小，patch_size为-1时，表示不切patch
            scale: lq和hq之间的分辨率比例，可用于超分数据集，scale=1时，表示lq和hq的分辨率相同
            flip: 选择flip
            rotation: 选择rotation
            read_mode: 数据集是lmdb还是disk，disk表示从磁盘中读取图像，lmdb表示实现制作成lmdb数据集
        """
        super(PairDataset, self).__init__(patch_size=patch_size, scale=scale)

        self.flip = flip
        self.rotation = rotation
        self.read_mode = read_mode
        self.hq_env, self.lq_env = None, None

        if read_mode.lower() == 'lmdb':
            self.hq_env, self.hq_keys = self.get_lmdb_info(hq_path)
            self.lq_env, self.lq_keys = self.get_lmdb_info(lq_path)
            assert self.hq_keys == self.lq_keys, f"read_mode: {read_mode}, hq_keys is not same as lq_keys"
        elif read_mode.lower() == "disk":
            self.hq_keys = self.get_disk_info(hq_path)
            self.lq_keys = self.get_disk_info(lq_path)
        else:
            raise ValueError(f"read_mode : {read_mode} is not support")

    def __len__(self):
        return len(self.lq_keys)

    def __getitem__(self, index):
        img_hq = self.get_img(keys=self.hq_keys, index=index, env=self.hq_env)
        img_lq = self.get_img(keys=self.lq_keys, index=index, env=self.lq_env)

        # bg2rgb and normalization
        # 是否进行通道转换需要自己根据具体的数据集进行判断
        img_hq = img_as_float32(img_hq[:, :, ::-1])
        img_lq = img_as_float32(img_lq[:, :, ::-1])

        # image2patch
        if self.patch_size > 0:
            img_lq, img_hq = self.random_img2patch(img_lq, img_hq)

        # image augmentation
        if self.flip or self.rotation:
            img_lq, img_hq = self.random_augment(img_lq, img_hq, flip=self.flip, rot=self.rotation)

        # image2tensor
        tensor_hq = img2tensor(img_hq)
        tensor_lq = img2tensor(img_lq)

        return {"hq": tensor_hq, "lq": tensor_lq}

