# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : dataset
# @Email : lianghao@whu.edu.cn

import os
import glob
import lmdb
import numpy as np

from tqdm import tqdm
from PIL import Image


class LMDB:
    def __init__(self, image_mode='png', read_func=None):
        if read_func is not None:
            self.read_func = read_func
        else:
            if image_mode in ['png', 'jpg', 'jpeg', 'bmp']:
                self.read_func = self._read_image_pil
            else:
                raise ValueError(f"Unknown image mode {image_mode},"
                                 f" only support 'png', 'jpg', 'jpeg', 'bmp' or input by yourself")

    def _read_image_pil(self, path):
        image = Image.open(path)
        return np.array(image)

    def get_total_buffer_size(self, paths):
        total_size = 0
        for path in tqdm(paths, desc='buffer'):
            data = self.read_func(path)
            total_size = total_size + data.nbytes
        return total_size

    def make_dataset(self, paths, save_dir, save_name):
        meta_info = []
        total_size = self.get_total_buffer_size(paths)
        lmdb_path = os.path.join(save_dir, save_name)
        lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 2)

        with lmdb_env.begin(write=True) as txn:
            for path in tqdm(paths, desc='lmdb'):
                _, file = os.path.split(path)
                data = self.read_func(path)
                key = file.encode('ascii')
                txn.put(key, data.tobytes())
                meta_info.append(file + '-' + f"{data.shape}")
        lmdb_env.close()

        with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as meta:
            for info in meta_info:
                meta.write(info + '\n')

        print(f"=>=> the filepath of ldmb is {lmdb_path}")


if __name__ == '__main__':
    paths = [path for path in glob.glob("~/CBSD68/*.png")]
    lmdb = LMDB(image_mode='png')
    lmdb.make_dataset(paths, save_dir="~/", save_name="cbsd68.lmdb")
