# -*- coding: utf-8 -*-
# @Time : 2024/01/23 19:33
# @Author : Liang Hao
# @FileName : pretrain_dncnn_unet.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
from src.utils import ARCH_REGISTRY
from .vdn_utils import UNet, DnCNN

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net


@ARCH_REGISTRY.register()
class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2, init=False):
        super(VDN, self).__init__()
        self.DNet = UNet(in_channels, in_channels, wf=wf, depth=dep_U, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels, dep=dep_S, num_filters=64, slope=slope)

        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if not m.bias is None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma
        

@ARCH_REGISTRY.register()
class VEMDNet(nn.Module):
    def __init__(self, pretrian_dncnn=True, pretrain_unet=True, pretrain=None):
        super().__init__()
        
        self.net_phi = UNet(3, 3, wf=64, depth=4, slope=0.2)
        self.net_theta = DnCNN(3, 3, dep=5, num_filters=64, slope=0.2)
        
        if pretrain is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if not m.bias is None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        else:
            dncnn_state, unet_state = {}, {}
            checkpoint = torch.load(pretrain, map_location=torch.device('cpu'))
            for key, value in checkpoint.items():
                if 'DNet' in key:
                    unet_state[key[5:]] = value
                else:
                    dncnn_state[key[5:]] = value
            
            if pretrian_dncnn:
                self.net_theta.load_state_dict(dncnn_state)
            
            if pretrain_unet:
                self.net_phi.load_state_dict(unet_state)

        
    
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
class VEMDNetConCat(nn.Module):
    def __init__(self, pretrian_dncnn=True, pretrain_unet=True, pretrain=None):
        super().__init__()
        
        self.net_phi = UNet(4, 3, wf=64, depth=4, slope=0.2)
        self.net_theta = DnCNN(3, 3, dep=5, num_filters=64, slope=0.2)
        
        if pretrain is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if not m.bias is None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        else:
            dncnn_state, unet_state = {}, {}
            checkpoint = torch.load(pretrain, map_location=torch.device('cpu'))
            for key, value in checkpoint.items():
                if 'DNet' in key:
                    unet_state[key[5:]] = value
                else:
                    dncnn_state[key[5:]] = value
            
            if pretrian_dncnn:
                self.net_theta.load_state_dict(dncnn_state)
            
            # if pretrain_unet:
            #     self.net_phi.load_state_dict(unet_state)

        
    
    def forward(self, x, step='E'):
        if step.upper() == 'E':
            with torch.no_grad():
                est_sigma = self.net_theta(x)
                est_sigma = est_sigma.detach()
                
            est_x = self.net_phi(torch.cat([torch.mean(est_sigma, dim=1, keepdim=True), x], dim=1))
            return est_x, est_sigma
        
        elif step.upper() == 'M':
            est_sigma = self.net_theta(x)
            with torch.no_grad():
                est_x = self.net_phi(torch.cat([torch.mean(est_sigma, dim=1, keepdim=True), x], dim=1))
                est_x = est_x.detach()
            return est_x, est_sigma
        
        elif step.upper() == 'PM':
            est_sigma = self.net_theta(x)
            return est_sigma
        