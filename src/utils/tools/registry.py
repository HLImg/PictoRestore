# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:49
# @Author : Liang Hao
# @FileName : registry.py
# @Email : lianghao@whu.edu.cn

class Registry(object):
    def __init__(self, name):
        """
        Based on https://github.com/XPixelGroup/BasicSR
        :param name:
        """
        self._name = name
        self._obj_dict = {}

    def _register(self, obj_name, obj):
        assert (obj_name.lower() not in self._obj_dict), (
            f"The object named {obj_name} was already registered "
            f"in {self._name}")

        self._obj_dict[obj_name.lower()] = obj

    def register(self, obj=None):
        if obj is None:
            def wrapper(func_or_class):
                obj_name = func_or_class.__name__
                self._register(obj_name, func_or_class)
                return func_or_class

            return wrapper

        obj_name = obj.__name__
        self._register(obj_name, obj)

    def get_obj(self, obj_name):
        ret = self._obj_dict.get(obj_name.lower(), None)
        if ret is None:
            print(f"No object named '{obj_name}' "
                  f"fount in module named '{self._name}' registry!")

        return ret

    def __contains__(self, obj_name):
        return obj_name in self._obj_dict

    def keys(self):
        return self._obj_dict.keys()



DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry('net_arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')