# -*- coding:utf-8 -*-
"""
@file name  : Conv2d-MaxPool2d.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 复写于 2024-12-18
@brief      : 卷积层和池化层的使用练习，直观感受卷积和平均池化的作用和效果
"""

import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from tools.common_tools import set_seed, transform_invert

# 设置随机种子，确保随机数列可重复
set_seed(3)

# load img
img = Image.open("./图1.png").convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)  # 添加 batch 维度

# create convolution layer
conv_layer = nn.Conv2d(3, 1, 3)

# 初始化 convolution layer 权重
nn.init.xavier_normal_(conv_layer.weight.data)  # 这里会用到前面用随机种子生成的随机数列

# create maxpool layer
avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2))  # 步长通常与 kernel_size 一致，确保池化时不重叠

# calculation
img_conv = conv_layer(img_tensor)
img_conv_pool = avgpool_layer(img_conv)

# visualization
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
print("池化前尺寸:{}\n池化后尺寸:{}".format(img_conv.shape, img_conv_pool.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_conv_pool = transform_invert(img_conv_pool[0, 0:3, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(131).imshow(img_raw)
plt.subplot(132).imshow(img_conv)
plt.subplot(133).imshow(img_conv_pool)
plt.show()
