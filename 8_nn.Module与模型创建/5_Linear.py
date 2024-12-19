# -*- coding:utf-8 -*-
"""
@file name  : Conv2d-MaxPool2d.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 复写于 2024-12-18
@brief      : 线性层使用练习
"""

import torch
import torch.nn as nn

inputs = torch.tensor([[1, 2, 3]], dtype=torch.float32)  # 前后张量的数据类型必须一致，显式声明是必要的
linear_layer = nn.Linear(3, 4)
linear_layer.weight.data = torch.rand(4, 3)
linear_layer.bias.data.fill_(0.5)
output = linear_layer(inputs)

print(output, output.shape)
