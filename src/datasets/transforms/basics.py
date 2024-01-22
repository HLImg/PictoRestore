# -*- coding: utf-8 -*-
# @Time    : 12/24/23 2:54 PM
# @File    : basics.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn
from __future__ import annotations

import math
import random

import torch
import cv2 as cv
import numpy as np

from skimage import img_as_float32
from torchvision.utils import make_grid



class BasicObject:
    def return_list(self, results: list) -> [list | object]:
        if len(results) == 1:
            return results[0]
        else:
            return results
    
    def uint2float32(self, image):
        return img_as_float32(image)

    def float32_to_uint8(self, image):
        return np.uint8((image.clip(0, 1) * 255.).round())

    def float32_to_uint16(self, image):
        return np.uint16((image.clip(0, 1) * 65535.).round())


class Compose(BasicObject):
    def __init__(self, transforms: list, p=1, k=-1):
        """
        :param transforms: [operator_1, operator_2, ..., operator_N]
        :param p: if p equals 1, transforms are applied sequentially. Otherwise, when random.random() > p,
                  transforms are applied in random order.
        :param k: When random.random() is greater than p, k operators will be randomly selected from the transforms.
        """
        self.transforms = transforms
        self.p = p
        if k <= 0:
            self.k = len(self.transforms)
        else:
            self.k = min(k, len(self.transforms))

    def __call__(self, *images):
        if self.p == 1 or random.random() < self.p:
            for t in self.transforms:
                images = t(*images)
                if not isinstance(images, (tuple, list)):
                    images = [images]
        else:
            transforms = random.sample(self.transforms, k=self.k)
            for t in transforms:
                images = t(*images)

        return self.return_list(images)


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
            image = image.copy()
            res.append(torch.from_numpy(image))

        return self.return_list(res)


class ToNdarray(BasicObject):
    def __call__(self, *tensors):
        res = []
        for tensor in tensors:
            tensor = tensor.float().detach().cpu().numpy()
            res.append(tensor)
        return self.return_list(res)


class ToUint8(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.float32_to_uint8(image)
            res.append(image)
        return self.return_list(res)


class Uint8ToSingle(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.uint2float32(image)
            res.append(image)
        return self.return_list(res)


class Uint16ToSingle(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.uint2float32(image)
            res.append(image)
        return self.return_list(res)


class ToUint16(BasicObject):
    def __call__(self, *images):
        res = []
        for image in images:
            image = self.float32_to_uint16(image)
            res.append(image)
        return self.return_list(res)


class ToImage(BasicObject):
    def __init__(self, out_type=np.uint8, min_max=(0, 1), rgb2bgr=True):
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
        self.rgb2bgr = rgb2bgr

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
                if self.rgb2bgr:
                    img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
            elif n_dim == 3:
                img_np = np.transpose(tensor.numpy(), (1, 2, 0))
                if img_np.shape[2] == 1:
                    img_np = np.squeeze(img_np, axis=2)
                else:
                    if self.rgb2bgr:
                        img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
            elif n_dim == 2:
                img_np = tensor.numpy()
            else:
                raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')

            if self.out_type == np.uint8:
                img_np = (img_np * 255.0).round()
            img_np = img_np.astype(self.out_type)

            res.append(img_np)

        return self.return_list(res)


class CenterCrop(BasicObject):
    def __init__(self, size: [int | list | tuple]):
        """
        Center-crop the input image by using its midpoint as a reference.
        The dimensions of the input images should either be (h, w, c) or (h, w)
        """
        if isinstance(size, int):
            self.shape = (size, size)
        elif isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise TypeError(
                    f"Expected the input tuple or list to have a length a 2, but received length: {len(size)}")
            else:
                self.shape = size
        else:
            raise TypeError(f"The input size must be of type tuple, list, or int. Received: {type(size)}")

    def __call__(self, *images):
        res = []
        for image in images:
            height, width = image.shape[:2]
            crop_h, crop_w = self.shape
            top = max((height - crop_h) // 2, 0)
            left = max((width - crop_w) // 2, 0)
            bottom = min(top + crop_h, height)
            right = min(left + crop_w, width)

            res.append(image[top: bottom, left: right])

        return self.return_list(res)


class RandomCrop(BasicObject):
    def __init__(self, size: [int | list | tuple], down_scale=1):
        """
        :param size: the size of cropped patches
        :param down_scale: By default, the first image is of high resolution. The images are
        organized in pairs by resolution, following the order [im_1_hr, im_2_hr, ..., im_i_lq, im_i+1_lq, ...]
        """
        if isinstance(size, int):
            self.shape = (size, size)
        elif isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise TypeError(
                    f"Expected the input tuple or list to have a length a 2, but received length: {len(size)}")
            else:
                self.shape = size
        else:
            raise TypeError(f"The input size must be of type tuple, list, or int. Received: {type(size)}")

        self.down_scale = down_scale

    def __call__(self, *images):
        if self.shape[0] < 0:
            return images
        n = len(images)
        max_h, max_w = images[0].shape[:2]
        if n > 1:
            min_h, min_w = images[1].shape[:2]
        else:
            min_h, min_w = images[0].shape[:2]

        ind_h = random.randint(0, max(min(min_h, max_h // self.down_scale), self.shape[0]) - self.shape[0])
        ind_w = random.randint(0, max(min(min_h, max_w // self.down_scale), self.shape[1]) - self.shape[1])

        res = []

        for i in range(0, n):
            h, w = images[i].shape[:2]
            if h == max_h and w == max_w:
                res.append(
                    images[i][ind_h * self.down_scale: (ind_h + self.shape[0]) * self.down_scale,
                    ind_w * self.down_scale: (ind_w + self.shape[1]) * self.down_scale]
                )
            else:
                res.append(images[i][ind_h: ind_h + self.shape[1], ind_w: ind_w + self.shape[1]])

        return self.return_list(res)


class Identity(BasicObject):
    def __call__(self, *images):
        return self.return_list(images)


class ToNdarray_chw2hwc(BasicObject):
    def __call__(self, *images):
        if isinstance(images[0], torch.Tensor):
            images = ToNdarray()(*images)

        if not isinstance(images, (list, tuple)):
            images = [images]
            
        if not isinstance(images[0], np.ndarray):
            raise TypeError(f"Only support torch.Tensor and np.ndarray, but received {type(images[0])}")
        
        res = []
        for image in images:
            if len(image.shape) == 4:
                res.append(np.transpose(image, (0, 2, 3, 1)))
            elif len(image.shape) == 3:
                res.append(np.transpose(image, (1, 2, 0)))
        return self.return_list(res)