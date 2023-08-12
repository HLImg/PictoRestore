# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 21:49
# @File    :   test.py
# @Email   :   lianghao@whu.edu.cn

import cv2
import torch
import numpy as np
from scipy.io import savemat
from source.net.denoise import NAFNet
from source.utils.image.transpose import tensor2img
from source.metric.image_metric import calculate_psnr, calculate_ssim
from source.dataset.common.pair_data import PairDataset
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
# from source.metric.cal_ssim import calculate_ssim



lq_path = '/data/dataset/sidd/lmdb/val_noisy.lmdb'
hq_path = '/data/dataset/sidd/lmdb/val_gt.lmdb'
pretrained_model = 'exp/denoise/NAFNet/0810_14_01/best_ckpt/best_iter_254000.ckpt'

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

# gt_mat = {}
# predict_mat = {}

print("start test")
for index in range(dataset.__len__()):
    res = dataset.__getitem__(index)
    lq = res['lq'].cuda().unsqueeze(0)
    hq = res['hq'].cuda().unsqueeze(0)
    
    predict = model(lq)
    
    # gt_mat["gt_" + str(index)] = tensor2img(hq)
    # predict_mat["pre_" + str(index)] = tensor2img(predict)
    psnr_1 += calculate_psnr(tensor2img(predict), tensor2img(hq), crop_border=0)
    ssim_1 += calculate_ssim(tensor2img(predict), tensor2img(hq), crop_border=0)

print(f"basicsr psnr :  {psnr_1 / dataset.__len__()}, ssim : {ssim_1 / dataset.__len__()}")


# savemat("hq.mat", gt_mat)
# savemat("predict.mat", predict_mat)


print("Finish")