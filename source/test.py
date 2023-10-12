# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 21:49
# @File    :   test.py
# @Email   :   lianghao@whu.edu.cn

import os
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from .net import Network
from .dataset import DataSet
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr


class HSITest:
    def __init__(self, config):
        self.config = config['test']
        self.config['net'] = config['net']
        
        
        self.dir = self.config['dir']
        self.mat = self.config['is_mat']
        self.is_show = self.config['is_show']
        self.save = self.config['is_save']
        
        self.clip = self.config['clip']
        self.bands = self.config['bands']
        
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        
        if self.save:
            self.save_denoised = os.path.join(self.dir, 'denoised')
            self.save_noised = os.path.join(self.dir, 'noised')
            self.save_diff = os.path.join(self.dir, 'diff')
            
            for path in [self.save_denoised, self.save_noised, self.save_diff]:
                if not os.path.exists(path):
                    os.mkdir(path)
            
        self.net = Network(self.config)()
        self.dataset = DataSet(self.config)()['test']
        
        if self.config['is_cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        ckpt = torch.load(self.config['ckpt'], map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
    
    def __call__(self):
        gts = []
        noiseds = []
        denoiseds = []
        
        self.net = self.net.to(self.device)
        
        self.net.eval()
        psnr, ssim = {}, {}
        num = self.dataset.__len__()
        for i in tqdm(range(num)):
            path = self.dataset.hq_keys[i].decode()
            name, _ = os.path.splitext(path)
            
            data = self.dataset.__getitem__(i)
            data = self.to_device(data)
            
            predict = self.net(data['lq'].unsqueeze(0))
            
            gts.append(self.tensor2hsi(data['hq']))
            noiseds.append(self.tensor2hsi(data['lq']))
            denoiseds.append(self.tensor2hsi(predict))
            
            
            metric_res = self.metric4tensor(data['hq'], predict)
            
            psnr[name] = metric_res['psnr']
            ssim[name] = metric_res['ssim']
            
            if self.save or self.is_show:
                gt_rgb = self.hsi2rgb(data['hq'])
                noised_rgb = self.hsi2rgb(data['lq'])
                denoised_rgb = self.hsi2rgb(predict)
                
                cv.imwrite(os.path.join(self.save_noised, name + '.png'), noised_rgb)
                cv.imwrite(os.path.join(self.save_denoised, name + '.png'), denoised_rgb)
                self.show(gt_rgb, noised_rgb, denoised_rgb, 
                          save_path=os.path.join(self.save_diff, name + '.png'), title=name)
            
                if self.is_show:
                    plt.show()
        
        total_psnr, total_ssim = 0.0, 0.0
        with open(os.path.join(self.dir, 'res.txt'), 'w') as file:
            for name in psnr.keys():
                file.write('{:<40}'.format(name) + ' ' * 4 + f'{psnr[name] : .4f} dB ' + ' ' * 4 + f'{ssim[name] : .4f} \n')
                total_psnr += psnr[name]
                total_ssim += ssim[name]
            
            file.write(f'avg psnr : {total_psnr / num :.4f},  avg ssim : {total_ssim / num : .4f}\n')
        
        print(f'avg psnr : {total_psnr / num :.4f},  avg ssim : {total_ssim / num : .4f}')
                
        
            

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        else:
            return data.to(self.device)
            
    
    def metric4tensor(self, gt, predict):
        
        res = {'psnr': 0., 'ssim': 0.}
        if len(gt.shape) < 4:
            gt = gt.unsqueeze(0)
        if len(predict.shape) < 4:
            predict = predict.unsqueeze(0)
            
        for i in range(gt.shape[0]):
            gt = gt[i].data.cpu().numpy().squeeze()
            predict = predict[i].data.cpu().numpy().squeeze()
        
            hsi_gt = np.transpose(gt, (1, 2, 0))
            hsi_predict = np.transpose(predict, (1, 2, 0))
            
            if self.clip:
                hsi_gt = hsi_gt.clip(0, 1)
                hsi_predict = hsi_predict.clip(0, 1)
            
            res['psnr'] += cal_psnr(hsi_gt, hsi_predict)
            res['ssim'] += cal_ssim(hsi_gt, hsi_predict, channel_axis=2)
            
        
        return res
    
    
    def hsi2rgb(self, hsi_chw):
        data = hsi_chw.data.cpu().numpy().squeeze()
        data = np.transpose(data, (1, 2, 0))
        image = data[:, :, tuple(self.bands)]
        image = img_as_ubyte(image.clip(0, 1))
        return image
    
    def tensor2hsi(self, tensor):
        tensor = tensor.data.cpu().numpy().squeeze()
        image = np.transpose(tensor, (1, 2, 0))
        return image
    
    
    def show(self, gt, noised, denoised, save_path, title):
        plt.subplot(231), plt.imshow(gt), plt.axis('off'), plt.title('gt')
        plt.subplot(232), plt.imshow(noised), plt.axis('off'), plt.title('noisy')
        plt.subplot(233), plt.imshow(denoised), plt.axis('off'), plt.title('denoised')
        
        plt.subplot(235), plt.imshow(np.abs(noised - gt)), plt.axis('off'), plt.title('noise')
        plt.subplot(236), plt.imshow(np.abs(denoised - gt)),  plt.axis('off'), plt.title('pre-noise')
        
        plt.suptitle(title)
        plt.savefig(save_path)