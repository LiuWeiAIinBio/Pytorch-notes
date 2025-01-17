# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : tingsongyu
# @date       : 2019-09-16 10:08:00
# @brief      : 通用函数
"""


import torch
import random
import psutil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


def set_seed(seed=1):
    torch.manual_seed(seed)  # 为 CPU 设置随机种子
    torch.cuda.manual_seed(seed)  # 为 GPU 设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多块 GPU，应该调用这个函数
    np.random.seed(seed)  # 为 numpy 设置随机种子
    random.seed(seed)  # 为 python 自带的 random 模块设置随机种子


def get_memory_info():
    virtual_memory = psutil.virtual_memory()
    used_memory = virtual_memory.used/1024/1024/1024
    free_memory = virtual_memory.free/1024/1024/1024
    memory_percent = virtual_memory.percent
    memory_info = "Usage Memory：{:.2f} G，Percentage: {:.1f}%，Free Memory：{:.2f} G".format(
        used_memory, memory_percent, free_memory)
    return memory_info







