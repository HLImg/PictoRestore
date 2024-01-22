# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:48
# @Author : Liang Hao
# @FileName : hsi.py
# @Email : lianghao@whu.edu.cn


import h5py
import numpy as np

from scipy.io import loadmat
from skimage.io import imread


def load_hsi_ICVL(path):
    """
     "Sparse Recovery of Hyperspectral Signal from Natural RGB Images"
     https://paperswithcode.com/dataset/icvl-hsi
    :param path:
    :return:
    """
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        mat_data = {key: f[key][()] for key in keys}
    return mat_data


def load_hsi_Realistic(path, ratio=1.):
    """
    Hyperspectral Image Denoising with Realistic Data
    @InProceedings{Zhang_2021_ICCV,
        author    = {Zhang, Tao and Fu, Ying and Li, Cheng},
        title     = {Hyperspectral Image Denoising With Realistic Data},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {2248-2257}
    }
    :param path: tif path for realistic image
    :param ratio: The ratio for the input is 50., while the ratio for ground truth is 1.0
    :return:
    """
    image = imread(path)
    image = image.astype(np.float32)
    image = image / 4096.
    image = np.transpose(image, (1, 2, 0))
    return image * ratio
