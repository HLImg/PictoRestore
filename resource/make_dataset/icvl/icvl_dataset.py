# -*- coding: utf-8 -*-
# @Time    : 9/14/23 10:38 PM
# @File    : icvl_dataset.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import glob
import lmdb
import cv2 as cv

from tqdm import tqdm
from down_icvl_hsi import load_hsi

def get_total_file_buffer_size(paths):
    total_size = 0
    for i in tqdm(range(len(paths)), desc='buffer'):
        path = paths[i]
        data = load_hsi(path)
        total_size += data['rad'].nbytes
    return total_size


def get_path(dir, meta_info, mode='train'):
    paths = []
    with open(meta_info + f'/{mode}.txt', 'r') as file:
        for line in file.readlines():
            paths.append(os.path.join(dir, line.strip()))
    return paths


def make_lmdb(icvl_dir, meta_dir, mode='train', lmdb_path='/',):
    meta_info = []
    paths = get_path(icvl_dir, meta_dir, mode)
    total_size = get_total_file_buffer_size(paths)
    lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 2)

    with lmdb_env.begin(write=True) as txn:
        for i in tqdm(range(len(paths)), desc='making'):
            path = paths[i]
            _, file = os.path.split(path)
            name, type = os.path.splitext(file)
            data = load_hsi(path)

            key = name.encode('ascii')
            txn.put(key, data['rad'].tobytes())
            meta_info.append(name + ' ' + f"{data['rad'].shape}")
    lmdb_env.close()

    with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as meta:
        for info in meta_info:
            meta.write(info + '\n')

    print("finish")


if __name__ == '__main__':
    meta_dir = './meta_info'
    icvl_dir = '/home/Public/Train/denoise/HSI/ICVL'

    dataset = {
        # 'train': '/home/Public/Train/denoise/HSI/icvl_train.lmdb',
        # 'valid': '/home/Public/Train/denoise/HSI/icvl_valid.lmdb',
        'test': '/home/Public/Train/denoise/HSI/icvl_test.lmdb'
    }

    for mode, lmdb_path in dataset.items():
        make_lmdb(icvl_dir, meta_dir=meta_dir, mode=mode, lmdb_path=lmdb_path)
