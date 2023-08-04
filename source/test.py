# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 21:49
# @File    :   test.py
# @Email   :   lianghao@whu.edu.cn

import cv2
import torch
import numpy as np
from source.net.denoise import NAFNet
from source.utils.image.transpose import tensor2img
from source.metric.image_metric import calculate_psnr
from source.dataset.common.pair_data import PairDataset
from skimage.metrics import peak_signal_noise_ratio as cal_psnr

lq_path = ''
hq_path = ''
pretrained_model = ''

dataset = PairDataset(lq_path, hq_path, read_mode='lmdb')

model = NAFNet(3, 32, mid_blk_nums=12,
                    enc_blk_nums=[2, 2, 4, 8],
                    dec_blk_nums=[2, 2, 2, 2],
                    blk_name="nafnet",
                    blk_params={
                        "DW_Expand" : 2,
                        "FFN_Expand" : 2,
                        "drop_out_rate" : 0.
                    })

ckpt = torch.load(pretrained_model)
model.load_state_dict(ckpt['net'])
model = model.cuda()

psnr_1 = 0.0
psnr_2 = 0.0

for index in range(dataset.__len__()):
    res = dataset.__getitem__(index)
    lq = res['lq'].cuda()
    hq = res['hq'].cuda()
    predict = model(lq)

    psnr_1 += calculate_psnr(tensor2img(predict), tensor2img(hq))
    psnr_2 += cal_psnr(tensor2img(predict), tensor2img(hq))

print(f"basicsr : {psnr_1 / dataset.__len__()}, skimage : {psnr_2 / dataset.__len__()}")
