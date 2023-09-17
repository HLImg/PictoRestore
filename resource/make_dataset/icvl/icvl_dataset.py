# -*- coding: utf-8 -*-
# @Time    : 9/14/23 10:38 PM
# @File    : icvl_dataset.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import glob
import lmdb
import cv2 as cv
import numpy as np

from tqdm import tqdm
from scipy.io import savemat, loadmat
from down_icvl_hsi import load_hsi


def get_total_file_buffer_size(paths, is_h5py=False):
    total_size = 0
    for i in tqdm(range(len(paths)), desc='buffer'):
        path = paths[i]
        data = load_hsi(path, is_h5py=is_h5py)
        total_size += data['rad'].nbytes
    return total_size


def get_path(dir, meta_info, mode='train'):
    paths = []
    with open(meta_info + f'/{mode}.txt', 'r') as file:
        for line in file.readlines():
            paths.append(os.path.join(dir, line.strip()))
    return paths


def make_lmdb(paths, lmdb_path='/', is_h5py=True):
    """需要使用h5py读取mat数据"""
    meta_info = []
    total_size = get_total_file_buffer_size(paths, is_h5py)
    lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 2)

    with lmdb_env.begin(write=True) as txn:
        for i in tqdm(range(len(paths)), desc='making'):
            path = paths[i]
            _, file = os.path.split(path)
            data = load_hsi(path, is_h5py=is_h5py)
            rad_img = np.transpose(data['rad'], (1, 2, 0))
            key = file.encode('ascii')
            txn.put(key, rad_img.tobytes())
            meta_info.append(file + ' ' + f"{rad_img.shape}")
    lmdb_env.close()

    with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as meta:
        for info in meta_info:
            meta.write(info + '\n')

    print("finish")

def crop_hsi_data(paths, save_dir, patch_size=256, stride=32, is_h5py=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    count = 0
    for i in tqdm(range(len(paths)), desc='croping'):
        path = paths[i]
        _, file = os.path.split(path)
        name, _ = os.path.splitext(file)
        data = load_hsi(path, is_h5py=is_h5py)['rad']
        c, h, w = data.shape

        idx = 1
        for row in range(0, h - stride, stride):
            for col in range(0, w - stride, stride):
                if row + patch_size > h or col + patch_size > w:
                    continue
                patch = data[:, row: row + patch_size, col: col + patch_size]
                savemat(file_name=os.path.join(save_dir, f"{name}_{idx}.mat"), mdict={'rad': patch})
                count += 1
                idx += 1

    print('*************************************************************')
    print(f'the number of patches is {count}')


def crop_hsi_test_data(paths, save_dir, patch_size=512, is_hypy=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    count = 0
    for i in tqdm(range(len(paths)), desc='test-croping'):
        path = paths[i]
        _, file = os.path.split(path)
        name, _ = os.path.splitext(file)
        data = load_hsi(path, is_h5py=is_hypy)['rad']
        
        data = np.rot90(data, k=-1, axes=(1, 2))
        c, h, w = data.shape
        ind_h = h // 2 - (patch_size // 2)
        ind_w = w // 2 - (patch_size // 2)
        
        patch = data[:, ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
        savemat(file_name=os.path.join(save_dir, f'{name}_center.mat'), mdict={'rad': patch})
        count += 1
    
    print('*************************************************************')
    print(f'the number of patches is {count}')
    
                


if __name__ == '__main__':
    meta_dir = './meta_info'
    icvl_dir = '/home/Public/Train/denoise/HSI/ICVL'

    dataset = {
        'train': '/data/dataset/HSI/ICVL/icvl_train.lmdb',
        'valid': '/data/dataset/HSI/ICVL/icvl_valid.lmdb',
        'test': '/data/dataset/HSI/ICVL/icvl_test.lmdb'
    }

    # train_paths = get_path(icvl_dir, meta_dir, 'train')
    # crop_hsi_data(train_paths, save_dir='/data/dataset/HSI/ICVL/icvl_train', patch_size=256, stride=192)
    # train_paths = [path for path in glob.glob('/data/dataset/HSI/ICVL/icvl_train/*.mat')]
    # make_lmdb(train_paths, lmdb_path=dataset['train'], is_h5py=False)

    # valid_paths = get_path(icvl_dir, meta_dir, 'valid')
    # crop_hsi_data(valid_paths, save_dir='/data/dataset/HSI/ICVL/icvl_valid', patch_size=512, stride=512)
    # valid_paths = [path for path in glob.glob('/data/dataset/HSI/ICVL/icvl_valid/*.mat')]
    # make_lmdb(valid_paths, lmdb_path=dataset['valid'], is_h5py=False)

    test_paths = get_path(icvl_dir, meta_dir, 'test')
    crop_hsi_test_data(test_paths, save_dir='/data/dataset/HSI/ICVL/icvl_test', patch_size=512)
    test_paths = [path for path in glob.glob('/data/dataset/HSI/ICVL/icvl_test/*.mat')]
    make_lmdb(test_paths, lmdb_path=dataset['test'], is_h5py=False)
