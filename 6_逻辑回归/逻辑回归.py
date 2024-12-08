# -*- coding:utf-8 -*-
"""
@file name  : 逻辑回归.py
@author     : LiuWei https://github.com/LiuWeiAIinBio
@date       : 2024-12-8
@brief      : 使用 Pytorch 实现逻辑回归模型，使用 dataset 和 dataloader 加载数据集，数据集 ex2data1.txt 来自吴恩达机器学习课程。
"""

import torch
from tools.my_dataset import LogisticRegressionDataset
from torch.utils.data import DataLoader

# 数据
BATCH_SIZE = 10
train_dir = "./ex2data1.txt"
train_data = LogisticRegressionDataset(data_dir=train_dir)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 模型参数
w1 = torch.zeros(1, requires_grad=True)  # 权重初始化
w2 = torch.zeros(1, requires_grad=True)  # 权重初始化
b = torch.zeros(1, requires_grad=True)  # 偏置初始化

# 迭代训练
lr = 0.01
MAX_EPOCH = 100
optimizer = torch.optim.SGD([w1, w2, b], lr)  # 定义优化器
criterion = torch.nn.BCELoss()  # 定义损失函数

def forward(x1, x2, w1, w2, b):
    term = w1*x1 + w2*x2 + b
    y_pred = torch.sigmoid(-term)
    return y_pred

flag = True
for epoch in range(MAX_EPOCH):
    if flag:
        for data in train_loader:
            x1, x2, y = data
            y_pred = forward(x1, x2, w1, w2, b)

            # backward
            optimizer.zero_grad()  # 梯度清零
            loss = criterion(y_pred, y)
            loss.backward()

            # update weights
            optimizer.step()

            if loss.data.numpy() < 0.2:
                mask = y_pred.ge(0.5).float()
                correct = (mask == y).float().sum()
                acc = correct / len(y)
                print(f"第 {epoch} 个 epoch 的分类准确率为：{acc}；",
                      f"各项参数为：w1 为 {w1.data.numpy()}，w2 为 {w2.data.numpy()}，b 为 {b.data.numpy()}；",
                      f"损失为 {loss.data.numpy()}，总共迭代了 {epoch} 个 epoch。")
                flag = False
                break

        if epoch % 20 == 0:
            mask = y_pred.ge(0.5).float()  # 布尔张量.float()：将布尔张量转换为浮点数张量，其中True会被转换为1.0，False会被转换为0.0
            correct = (mask == y).float().sum()
            acc = correct / len(y)
            print(f"第 {epoch} 个 epoch 的分类准确率为：{acc}")
