# -*- coding:utf-8 -*-
"""
@file name  : Conv2d-MaxPool2d.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 复写于 2024-12-18
@brief      : 池化和反池化的使用练习
"""

import torch
import torch.nn as nn

# pooling
img_raw = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float32)
maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
img_pool, indices = maxpool_layer(img_raw)

# unpooling
maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
img_unpool = maxunpool_layer(img_pool, indices)

print("img_raw:\n{}\nimg_pool:\n{}\nimg_unpool:\n{}".format(img_raw, img_pool, img_unpool))
