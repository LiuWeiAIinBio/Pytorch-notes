# [Pytorch-notes](https://github.com/LiuWeiAIinBio/Pytorch-notes)

Pytorch 学习笔记的主要参考内容为：
- 深度之眼《深度学习PyTorch框架班》：视频课地址为 https://ai.deepshare.net/p/t_pc/course_pc_detail/video/v_5e9e5f6ddcef2_TCLvUDOF?product_id=p_5df0ad9a09d37_qYqVmt85&content_app_id=&type=6 ，代码地址为 https://github.com/JansonYuan/Pytorch-Camp ，一个比较详细的笔记 https://pytorch.zhangxiann.com/ 。
- 伊莱·史蒂文斯的《PyTorch深度学习实战》的第二章和第三章
- 龙良曲老师的《龙良曲PyTorch入门到实战》的课时 14 到课时 30，视频课地址为 https://www.bilibili.com/video/BV12B4y1k7b5?vd_source=2e0bed8f939119c48817ce61f4f75bdd&spm_id_from=333.788.videopod.episodes

<br/>

## [01. 使用pytorch运行预训练模型](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/01_%E4%BD%BF%E7%94%A8pytorch%E8%BF%90%E8%A1%8C%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
本部分笔记整理自伊莱·史蒂文斯的《PyTorch深度学习实战》的第二章。

## [02. 张量](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/02_%E5%BC%A0%E9%87%8F)
深入理解 PyTorch 张量整理自伊莱·史蒂文斯的《PyTorch深度学习实战》的第三章，PyTorch 张量的常用 API 整理自龙良曲老师的视频课程《龙良曲PyTorch入门到实战》的课时 14 到课时 30。

## [03. 动态计算图和自动求导](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/03_%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E5%9B%BE%E5%92%8C%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC)
动态计算图是 PyTorch 的基础，对理解整个 PyTorch 框架非常重要。

## [04. 数据读取机制Dataloader与Dataset](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/04_%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E6%9C%BA%E5%88%B6Dataloader%E4%B8%8EDataset)
建立一个深度学习项目，第一步是准备数据，数据怎样传递给模型用于训练呢，就是通过 Dataloader 与 Dataset 实现的。

## [05. 单变量线性回归](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/05_%E5%8D%95%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92)
本部分是我使用 Pytorch 实现单变量线性回归模型，非视频课笔记，主要是练习自定义 dataset 子类，及使用 dataset 和 dataloader 加载数据集，dataset 子类定义在 ./tools/my_dataset.py 中，数据集 ex1.txt 来自吴恩达机器学习课程。

## [06. 逻辑回归](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/06_%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92)
本部分是我使用 Pytorch 实现逻辑回归模型，非视频课笔记，主要是练习自定义 dataset 子类，及使用 dataset 和 dataloader 加载数据集，dataset 子类定义在 ./tools/my_dataset.py 中，数据集 ex2data1.txt 来自吴恩达机器学习课程。

## [07. transforms模块机制和数据预处理方法](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/07_transforms%E6%A8%A1%E5%9D%97%E6%9C%BA%E5%88%B6%E5%92%8C%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95)
transforms 是图片操作工具集，如果处理的数据是图片，需要在训练文件中定义具体的 transforms 方法，使用该方法对图片数据进行处理。

## [08_nn.Module与模型创建](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/08_nn.Module%E4%B8%8E%E6%A8%A1%E5%9E%8B%E5%88%9B%E5%BB%BA)
建立一个深度学习项目，第一步是准备数据，第二步就是基于 nn.Module 创建模型。本部分的 .ipynb 文件详细地介绍了 Pytorch 的 nn.Module，.py 文件是 nn 模块的卷积层、池化层、全连接层、激活函数层等网络层的使用练习。

## [09_模型训练](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/09_%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
建立一个深度学习项目，第一步是准备数据，第二步是创建模型，第三步就是训练模型。在迭代训练模型之前，我们需要初始化模型的权值，选择损失函数，设置优化器和设置学习率调整策略，然后加载数据到模型中进行前向传播和反向传播更新参数，训练模型。本部分介绍了权值初始化方法、损失函数、优化器和学习率调整策略。

## [10_正则化Regularization](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/10_%E6%AD%A3%E5%88%99%E5%8C%96Regularization)
过拟合是训练模型时的一种常见问题，我们可通过设置正则化手段来抑制过拟合，本部分介绍了三种正则化手段：
- 在优化器中设置正则项
- 在模型中设置 Dropout 层
- 在模型中设置 Normalization 层

## [11_可视化工具](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/11_%E5%8F%AF%E8%A7%86%E5%8C%96%E5%B7%A5%E5%85%B7)
本部分介绍了三种模型训练时的可视化工具：
- 训练模型时需要使用可视化工具监控训练情况，不对劲时就要及时停止，TensorBoard 可以帮助我们可视化训练过程。
- `torchsummary()` 能够查看模型的输入和输出的形状，便于我们调试。
- 由于 PyTorch 是基于动态图实现的，因此在一次迭代运算结束后，一些中间变量如非叶子节点的梯度和特征图，会被释放掉，如果想要获取这些中间变量，可以使用 Hook 函数。

## [12_模型的其他操作](https://github.com/LiuWeiAIinBio/Pytorch-notes/tree/main/12_%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%85%B6%E4%BB%96%E6%93%8D%E4%BD%9C)
本部分介绍了在训练模型时，以及模型训练好之后，对模型的操作：
- 模型的保存、加载和断点续训练
- 模型 Finetune
- 使用 GPU 训练模型
