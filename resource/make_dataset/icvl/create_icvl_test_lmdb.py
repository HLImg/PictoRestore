# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/21 14:04:04
# @FileName:  create_icvl_test_lmdb.py
# @Contact :  lianghao@whu.edu.cn

import os
import glob
import lmdb
import cv2 as cv
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat, savemat

def get_total_file_buffer_size(paths):
    total_size_gt = 0
    total_size_inp = 0
    for i in tqdm(range(len(paths)), desc='buffer'):
        data = loadmat(paths[i])
        gt = np.float32(data['gt'])
        inp = np.float32(data['input'])
        total_size_gt += gt.nbytes
        total_size_inp += inp.nbytes
    
    return {'inp': total_size_inp, 'gt': total_size_gt}


def make_lmdb(paths, lmdb_path, mode, total_size ):
    meta_info = []    

    lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 2)

    with lmdb_env.begin(write=True) as txn:
        for i in tqdm(range(len(paths)), desc=f'make-{mode}'):
            _, file = os.path.split(paths[i])
            name, _ = os.path.splitext(file)
            
            data = loadmat(paths[i])
            
            shape = data[mode].shape
            image = np.transpose(np.float32(data[mode]), (2, 0, 1))
            
            key = name.encode('ascii')
            txn.put(key, image.tobytes())
            meta_info.append(name + f" ({shape[0]},{shape[1]},{shape[2]})")
    
    lmdb_env.close()
    
    with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as meta:
        for info in meta_info:
            meta.write(info + '\n')
    

def make_test_lmdb(data_dir, lmdb_dir='/'):
    paths = [path for path in glob.glob(data_dir)]
    
    total_size = get_total_file_buffer_size(paths)
    
    lmdb_path_gt = lmdb_dir + '_hq.lmdb'
    lmdb_path_inp = lmdb_dir + '_lq.lmdb'
    
    make_lmdb(paths, lmdb_path_gt, mode='gt', total_size=total_size['gt'])
    make_lmdb(paths, lmdb_path_inp, mode='input', total_size=total_size['inp'])
    
    print('finish')

if __name__ == '__main__':
    modes = ["noniid", 'impulse', 'mix', 'stripe', 'deadline']
    
    for mode in modes:
        data_dir = f'/data/dataset/hsi/icvl_test/512_{mode}/*.mat'
        lmdb_dir = f'/data/dataset/hsi/complex_test/sert_{mode}'
        make_test_lmdb(data_dir, lmdb_dir)