# -*- coding: utf-8 -*-
# @Time : 2024/01/23 13:45
# @Author : Liang Hao
# @FileName : backbone.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .noise_est import RIRNet
from src.arches.denoising.spatial_channel import SCTUnet

from src.utils import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class VEMIDNet(nn.Module):
    def __init__(self, 
                 noise_est_param,
                 noise_rm_param):
        super().__init__()
        
        self.net_theta = RIRNet(**noise_est_param)
        self.net_phi = SCTUnet(**noise_rm_param)
        
    def forward(self, x, step='E'):
        if step.upper() == 'E':
            with torch.no_grad():
                est_sigma = self.net_theta(x)
                est_sigma = est_sigma.detach()
            est_x = self.net_phi(est_sigma + x)
            return est_x, est_sigma
        elif step.upper() == 'M':
            est_sigma = self.net_theta(x)
            with torch.no_grad():
                est_x = self.net_phi(est_sigma + x)
                est_x = est_x.detach()
            return est_x, est_sigma
        elif step.upper() == 'PM':
            est_sigma = self.net_theta(x)
            return est_sigma
        

@ARCH_REGISTRY.register()
class VEMIDNetPreTrain(nn.Module):
    def __init__(self, 
                 noise_est_param,
                 noise_rm_param):
        super().__init__()
        
        self.net_theta = RIRNet(**noise_est_param)
        self.net_phi = SCTUnet(**noise_rm_param)
        
    def forward(self, x):
        phi_x = self.net_phi(x)
        theta_sigma = self.net_theta(x)
        return phi_x, theta_sigma