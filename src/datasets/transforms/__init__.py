# -*- coding: utf-8 -*-
# @Time    : 12/24/23 2:47 PM
# @File    : __init__.py.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

"""
The difference between the rewritten transforms and the orginal
ones lies in the processing of ndarray data instead of PIL images.
In addition, our transforms are designed to support the input of
single or multiple images.
"""

from .basics import Compose

__all__ = [
    'Compose'
]