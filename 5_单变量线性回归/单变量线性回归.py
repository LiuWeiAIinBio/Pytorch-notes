# -*- coding:utf-8 -*-
"""
@file name  : 单变量线性回归.py
@author     : LiuWei https://github.com/LiuWeiAIinBio
@date       : 2024-12-4
@brief      : 使用 Pytorch 实现单变量线性回归模型，使用 dataset 和 dataloader 加载数据集，数据集 ex1.txt 来自吴恩达机器学习课程。
"""

import torch
from tools.my_dataset import linearRegressionDataset
from torch.utils.data import DataLoader

# 数据
BATCH_SIZE = 10
train_dir = "./ex1.txt"
train_data = linearRegressionDataset(data_dir=train_dir)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 模型参数
w = torch.randn(1, requires_grad=True)  # 权重初始化
b = torch.randn(1, requires_grad=True)  # 偏置初始化

# 迭代训练
lr = 0.01
MAX_EPOCH = 100
optimizer = torch.optim.SGD([w, b], lr)  # 定义优化器
criterion = torch.nn.MSELoss()  # 定义损失函数

def forward(x, w, b):
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    return y_pred

flag = True
for epoch in range(MAX_EPOCH):
    if flag:
        for data in train_loader:
            x, y = data
            y_pred = forward(x, w, b)

            # backward
            optimizer.zero_grad()  # 梯度清零
            loss = criterion(y_pred, y)
            loss.backward()

            # update weights
            optimizer.step()

            if loss.data.numpy() < 1:
                print(w.data.numpy(), b.data.numpy(), loss.data.numpy(), epoch)
                flag = False
                break
