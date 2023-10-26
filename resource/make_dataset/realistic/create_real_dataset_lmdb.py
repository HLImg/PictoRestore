# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/10/25 21:09:57
# @FileName:  create_real_dataset_lmdb.py
# @Contact :  lianghao@whu.edu.cn

import os
import glob
import lmdb
import numpy as np

from tqdm import tqdm
from skimage.io import imread

def load_tif(path, ratio=1.):
    image = imread(path)
    image = image.astype(np.float32)
    image = image / 4096.
    image = np.transpose(image, (1, 2, 0))
    return image * ratio

def get_total_file_buffer_size(paths, ratio=1.):
    total_size = 0
    for i in tqdm(range(len(paths)), desc='buffer'):
        data = load_tif(paths[i])
        total_size += data.nbytes
    
    return total_size

def get_image_paths(main_dir, meta_info):
    train_dir_gt, train_dir_inp = [], []
    test_dir_gt, test_dir_inp = [], []
    
    with open(meta_info, 'r') as file:
        test_data = [line.strip() for line in file.readlines()]
    
    for path in glob.glob(os.path.join(main_dir, 'gt/*.tif')):
        _, name = os.path.split(path)
        
        gt = os.path.join(main_dir, 'gt', name)
        input = os.path.join(main_dir, 'input50', name)
        
        if name in test_data:
            test_dir_gt.append(gt)
            test_dir_inp.append(input)
        else:
            train_dir_gt.append(gt)
            train_dir_inp.append(input)
    
    return {'train': 
                {'input':train_dir_inp, 'gt': train_dir_gt}, 
            'test': 
                {'input': test_dir_inp, 'gt': test_dir_gt}}


def make_lmdb(paths, lmdb_dir, ratio, mode='train'):
    meta_info = []
    total_size = get_total_file_buffer_size(paths, ratio)
    flag = 'lq' if ratio > 1 else 'hq'
    lmdb_path = os.path.join(lmdb_dir, f'{mode}_{flag}.lmdb')
    lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 2)
    
    with lmdb_env.begin(write=True) as txn:
        for i in tqdm(range(len(paths)), desc=f'make-{mode}'):
            _, file = os.path.split(paths[i])
            name, _ = os.path.splitext(file)
            
            data = load_tif(paths[i], ratio)
            shape = data.shape
            image = np.transpose(data, (2, 0, 1))
            key = name.encode('ascii')
            txn.put(key, image.tobytes())
            meta_info.append(name + f" ({shape[0]},{shape[1]},{shape[2]})")
    
    lmdb_env.close()
    
    with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as meta:
        for info in meta_info:
            meta.write(info + '\n')
    

def main(main_dir, meta_info, lmdb_dir):
    dirs = get_image_paths(main_dir, meta_info)
    
    # making train
    make_lmdb(dirs['train']['input'], lmdb_dir, ratio=50., mode='train')
    make_lmdb(dirs['train']['gt'], lmdb_dir, ratio=1., mode='train')
    
    # making test
    make_lmdb(dirs['test']['input'], lmdb_dir, ratio=50., mode='test')
    make_lmdb(dirs['test']['gt'], lmdb_dir, ratio=1., mode='test')


if __name__ == '__main__':
    main_dir = '/data/dataset/hsi/real_dataset'
    meta_info = './meta_info/test.txt'
    lmdb_dir = '/data/dataset/hsi/real_dataset'
    
    main(main_dir, meta_info, lmdb_dir)