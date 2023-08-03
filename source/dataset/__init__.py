# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/2 18:53
# @File    :   __init__.py.py
# @Email   :   lianghao@whu.edu.cn

class DataSet:
    def __init__(self, config):
        self.info = config["dataset"]
        self.dataset_task = self.info["task"]
        self.dataset_name = self.info["name"]
        self.dataset_params = self.info["param"]

    def __call__(self, *args, **kwargs):
        if self.dataset_task.lower() == 'denoise':
            from source.dataset.denoise import select_dataset
        else:
            raise ImportError(f"the task named {self.dataset_task} is not exits in dataset")

        return select_dataset(name=self.dataset_name, param=self.dataset_params)