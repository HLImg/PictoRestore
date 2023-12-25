# -*- coding: utf-8 -*-
# @Time    : 12/24/23 2:54 PM
# @File    : basics.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import math
import torch
import numpy as np

from torchvision.utils import make_grid


class BasicObject:
    def return_list(self, results: list) -> [list | object]:
        if len(results) == 1:
            return results[0]
        else:
            return results

    def uint8_to_single(self, image):
        return np.float32(image / 255.)

    def uint16_to_single(self, image):
        return np.float32(image / 65535.)

    def single2uint8(self, image):
        return np.uint8((image.clip(0, 1) * 255.).round())

    def single2uint16(self, image):
        return np.uint16((image.clip(0, 1) * 65535.).round())




class Compose(BasicObject):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *images):
        for t in self.transforms:
            images = t(*images)

        return images


class ToTensor(BasicObject):
    def __init__(self, inp_mode='hwc'):
        """
        convert image into tensor
        :param inp_mode: the channel's mode of input images.
        """
        self.mode = inp_mode

    def __call__(self, *images):
        res = []
        for image in images:
            if len(image.shape) == 2:
                image = image[np.newaxis, :]
            else:
                if self.mode.lower() == 'hwc':
                    image = np.transpose(image, (2, 0, 1))
            res.append(torch.from_numpy(image))

        return self.return_list(res)


class ToNdarray(BasicObject):
    def __call__(self, *tensors):
        res = []
        for tensor in tensors:
            tensor = tensor.float().detach().cpu()
            res.append(tensor)
        return self.return_list(res)


class ToUint8(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.single2uint8(image)
            res.append(image)
        return self.return_list(res)


class ToUint16(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.single2uint16(image)
            res.append(image)
        return self.return_list(res)


class ToImage(BasicObject):
    def __init__(self, out_type=np.uint8, min_max=(0, 1)):
        """
        convert torch tensors into image.
        acceptable shapes of converted tensor are :
            1) 4D mini-batch tensor of shape (batch_size, channels, height, width).
            2) 3D tensor of shape (channels, height, width).
            3) 2D tensor of shape (height, width).
        :param out_type: transform the output to the specified Numpy data type.
        :param min_max: min and max values for clamp.
        """
        self.out_type = out_type
        self.min_max = min_max

    def __call__(self, *tensors):
        res = []
        for tensor in tensors:
            if not torch.is_tensor(tensor):
                raise TypeError(f"tensor of *tensors expected expected, but got {type(tensor)}")

            tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*self.min_max)
            tensor = (tensor - self.min_max[0]) / (self.min_max[1] - self.min_max[0])

            n_dim = tensor.dim()
            if n_dim == 4:
                img_np = make_grid(tensor, normalize=False, nrow=int(math.sqrt(tensor.size(0))))
                img_np = np.transpose(img_np.numpy(), (1, 2, 0))
            elif n_dim == 3:
                img_np = np.transpose(tensor.numpy(), (1, 2, 0))
                img_np = img_np.squeeze()
            elif n_dim == 2:
                img_np = tensor.numpy()
            else:
                raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')

            if self.out_type == np.uint8:
                img_np = (img_np * 255.0).round()
            img_np = img_np.astype(self.out_type)

            res.append(img_np)

        return self.return_list(res)
