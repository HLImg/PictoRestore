# -*- coding: utf-8 -*-
# @Time    : 12/24/23 2:47 PM
# @File    : __init__.py.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

from common import *


if __name__ == '__main__':
    cf1 = dict(
        hq_path='D:/TestDataSet/CBSD68/gt.lmdb',
        down_scale=1,
        patch_size=120,
        aug_mode=None,
        read_mode='lmdb'
    )

    # dataset = registy_dataset.get("common")(**cf1)
    #
    # print(len(dataset))