# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 12:56
# @File    :   __init__.py
# @Email   :   lianghao@whu.edu.cn

def select_dataset(name, param):
    train_dataset, test_dataset = (None, None)

    if name.lower() == "pair_data":
        from source.dataset.common.pair_data import PairDataset as dataset
    else:
        raise ValueError(f"the name {name} is not exist in dataset-denoise")

    if 'train' in param.keys():
        train_dataset = dataset(**param["train"])
    if 'test' in param.keys():
        test_dataset = dataset(**param['test'])

    return {"train": train_dataset, "test": test_dataset}