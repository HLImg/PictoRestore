# -*- coding: utf-8 -*-
# @Time : 2023/12/28
# @Author : Liang Hao
# @FileName : __init__
# @Email : lianghao@whu.edu.cn

from .hyperspectral import (load_hsi_ICVL, load_hsi_Realistic)


__all__ = [
    'load_hsi_ICVL', 'load_hsi_Realistic'
]