# 1. 模型（module）构建

**1) 以实现一个 LeNet 为例介绍模型构建过程：**

模型构建分为两个步骤：构建网络层和拼接网络层

在下面 lenet.py 的代码中，首先在 `__init__()` 中构建网络层，然后在 `forward()` 前向传播函数中拼接网络层。

```
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    # 在 __init__() 中构建网络层
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    # 在 `forward()` 前向传播函数中拼接网络层
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

**2) nn.Module**

**所有的模型和网络层都是继承于 nn.Module 这个类**

**torch.nn 有 4 个主要部分：**
    
- nn.Parameter：张量子类，存储为张量的形式，同样也具有张量的 8 个基本属性，表示可学习参数，如权重、偏置等
- nn.Module：所有网络层基类，管理网络属性
- nn.functional：包含函数的具体实现，如卷积、池化、激活函数等
- nn.init：提供丰富的参数初始化方法

**nn.Module 有 8 个重要的属性，以 `OrderedDict()` 有序字典的形式存放其取值，先介绍 2 个比较重要的属性：**

    parameters：存储管理 nn.Parameter 类
    modules：存储管理 nn.Module 类
    buffers：存储管理缓冲属性
    ***hooks：存储管理钩子函数，*** 代表有 5 个类似的属性
    
    LeNet 继承于 nn.Module，所以 LeNet 类的实例具有 nn.Module 的 8 个重要属性，其中 modules 属性会在其有序字典中会存放前面构建网络层部分构建的网络层，这些网络层在构建时同样也是继承于 nn.Module，所以这些网络层也具有 nn.Module 的 8 个重要属性。即：一个 module 可以包含多个子 module，每个 module（module and 子 module）都有 8 个有序字典管理其属性。

**在前面模型构建中，需要先在 `__init__()` 中初始化 nn.Module 父类的 `__init__()`，实现 nn.Module 属性的初始化，即生成 8 个空的有序字典管理 module 的属性：**

```
class LeNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
    ···
```

<br/>

# 2. 模型容器 Containers

**1) nn.Sequetial**

**nn.Sequetial() 是网络层容器，用于按顺序包装多个网络层。**

以实现一个 LeNet 为例介绍 nn.Sequetial：在 LeNet 中，以全连接层为边界，使用 nn.Sequetial() 将前面的卷积层和池化层包装为特征提取模块(features)，使用 nn.Sequetial() 将后面的全连接层包装为分类器模块(classifier)，然后在前向传播函数中将两个模块连接起来，特征提取模块提取特征，分类器模块输出分类结果。

```
import torch.nn as nn


class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super().__init__()

        # 使用 nn.Sequetial() 将卷积层和池化层包装为特征提取模块，并赋给实例属性 features
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        # 使用 nn.Sequetial() 将 3 个全连接层包装为分类器模块，并赋给实例属性 classifier
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)

        # 在前向传播函数中将两个模块连接起来
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size()[0], -1)
            x = self.classifier(x)
            return x


# 实例化模型，并使用示例数据运行模型的前向传播函数
net = LeNetSequential(classes=2)
example_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
output = net(example_img)
print(net)
print(output)
```

**在将网络层包装成模块时，还可以引入有序字典 OrderDict，以备在需要的时候实现对模块内网络层的索引。**

```
import torch.nn as nn
from collections import OrderedDict


class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
```

**2) nn.ModuleList**

**nn.ModuleList() 是网络层容器，用于包装多个网络层，可以以迭代的方式创建多个网络层，也可以索引网络层，类似于 python 的 list。**

主要方法：

    append()：在 ModuleList 后面添加网络层
    extend()：拼接两个 ModuleList
    insert()：在 ModuleList 中指定位置插入网络层

```
class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x
```

**3) nn.ModuleDict**

**nn.ModuleDict 像 python 的 dict 一样包装多个网络层，以索引的方法调用网络层。**

主要方法：

    clear()：清空 ModuleDict
    items()：返回可迭代的键值对
    keys()：返回字典的键
    values()：返回字典的值
    pop()：从字典中弹出一对键值，该对儿键值从字典中删除

```
class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```

**4) 总结**

nn.Sequetial：顺序性，各网络层之间严格按照顺序，常用于模块构建，内部包含一个 for 循环前向传播机制，用于模块内网络层的前向传播

nn.ModuleList：迭代性，常用于大量重复网络构建，通过 for 循环实现重复构建

nn.ModuleDict：索引性，常用于可选择的网络层的构建

<br/>

# 3. 模型介绍：AlexNet

**AlexNet 在 2012 年开创了卷积神经网络的新时代，AlexNet 的特点如下：**

1. 采用 ReLU 激活函数替换 sigmoid 等饱和激活函数，减轻梯度消失。
2. 采用 LRN（local response normalization，局部响应值归一化）对数据归一化，减轻梯度消失。现在常用的是 batch normalization。
3. Dropout：提高全连接层的鲁棒性，增加网络的泛化能力。
4. Data Augmentation

<br/>

# 4. nn 网络层

**1) nn.Conv2d()**

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')`

函数功能：对多个二维信号进行二维卷积

主要参数：

    in_channels：输入通道数
    out_channels：输出通道数，等价于卷积核个数
    kernel_size：卷积核尺寸
    stride：步长
    padding：填充宽度，主要是为了调整输出的特征图大小，一般把 padding 设置合适的值后，保持输入和输出的图像尺寸不变
    dilation：卷积核扩张幅度，默认为 1，1 代表没有扩张；卷积核扩张主要是为了扩大视野
    groups：分组卷积设置，主要是为了模型的轻量化
    bias：偏置

**2) nn.MaxPool2d()**

`nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`

函数功能：进行 2 维的最大值池化

主要参数：

    kernel_size：池化核尺寸
    stride：步长，通常与 kernel_size 一致，确保池化时不重叠
    padding：填充宽度，主要是为了调整输出的特征图大小，一般把 padding 设置合适的值后，保持输入和输出的图像尺寸不变
    dilation：池化核扩张幅度，默认为 1，1 代表没有扩张
    ceil_mode：默认为 False，尺寸向下取整；为 True 时，尺寸向上取整
    return_indices：为 True 时，返回最大值池化所使用的像素的索引，这些记录的索引通常在反最大值池化时使用，把像素放在对应的位置上

**3) nn.AvgPool2d()**

`torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)`

函数功能：进行 2 维的平均池化

主要参数：

    kernel_size：池化核尺寸
    stride：步长，通常与 kernel_size 一致，确保池化时不重叠
    padding：填充宽度，主要是为了调整输出的特征图大小，一般把 padding 设置合适的值后，保持输入和输出的图像尺寸不变
    dilation：池化核扩张幅度，默认为 1，1 代表没有扩张
    ceil_mode：默认为 False，尺寸向下取整；为 True 时，尺寸向上取整
    count_include_pad：在计算平均值时，是否把 padding 填充值考虑在内计算
    divisor_override：除法因子。在计算平均值时，分子是像素值的总和，分母默认是像素值的个数；如果设置了 divisor_override，把分母改为 divisor_override

**4) nn.Linear()**

`nn.Linear(in_features, out_features, bias=True)`

函数功能：线性层又称为全连接层，其每个神经元与上一个层所有神经元相连，实现对前一层的线性组合或线性变换。

主要参数：

    in_features：输入结点数
    out_features：输出结点数
    bias：是否需要偏置

**5) nn.Sigmoid()**

激活函数层

计算公式：$y=\frac{1}{1+e^{-x}}$

导数/梯度公式：$y^{\prime}={y}*{(1-y)}$

特性：

    输出值在(0,1)，符合概率
    导数/梯度范围是 [0, 0.25]，容易导致梯度消失
    输出为非 0 均值，破坏数据分布

**6) nn.tanh()**

激活函数层

计算公式：$y=\frac{\sin x}{\cos x}=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$

导数/梯度公式：$y^{\prime}=1-y^{2}$

特性：

    输出值在(-1, 1)，数据符合 0 均值
    导数/梯度范围是 (0,1)，梯度消失麻烦比 Sigmoid 小一些，但依然容易导致梯度消失

**7) nn.ReLU()**

激活函数层，修正线性单元

计算公式：$y=max(0, x)$，在正半轴为 $y=x$，在负半轴恒为 0

导数/梯度公式：$y^{\prime}=1(x>0), undefined(x=0), 0(x<0)$

特性：

    输出值均为正数，负半轴的导数为 0，容易导致死神经元
    导数/梯度是 1，缓解梯度消失，但容易引发梯度爆炸

**针对 RuLU 负半轴的导数为 0，会导致死神经元的缺点，介绍 3 种改进的 RuLU 激活函数：**

- nn.LeakyReLU()：负半轴在第三象限有一个很小的倾斜，有一个参数 negative_slope 设置负半轴斜率

- nn.PReLU()：有一个参数 init 设置初始斜率，这个斜率是可学习的
    
- nn.RReLU()：R 是 random 的意思，负半轴斜率每次都是随机取 `[lower, upper]` 之间的一个数

<br/>

**梯度消失**：由于计算导数的链式法则，小于 1 的导数在多次相乘后得到的值越来越小，直至接近于 0，这就是导数消失/梯度消失

**梯度爆炸**：由于计算导数的链式法则，大于 1 的导数在多次相乘后得到的值越来越大，这就是导数爆炸/梯度爆炸
