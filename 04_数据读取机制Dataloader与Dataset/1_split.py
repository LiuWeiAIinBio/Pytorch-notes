# -*- coding:utf-8 -*-
"""
# @file name  : 1_split.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 复写于 2024-12-2
# @brief      : 将数据集划分为训练集，验证集，测试集
"""


import os
import sys
import random
import shutil


"""
__file__ 返回当前执行文件的绝对路径（目录部分 + 文件名）
os.path.abspath() 返回绝对路径
os.path.dirname() 返回绝对路径的目录部分
BASE_DIR 变量的值是当前执行文件的绝对路径的目录部分
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    # 构建路径
    dataset_dir = os.path.abspath(os.path.join(BASE_DIR, "RMBdata"))
    split_dir = os.path.abspath(os.path.join(BASE_DIR, "RMBdata_split"))
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    if not os.path.exists(dataset_dir):
        raise Exception("{} 不存在，请下载 RMBdata 放到 {} 下".format(dataset_dir, os.path.dirname(dataset_dir)))

    # 设置分割比例
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)  # 打乱顺序
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            if img_count == 0:
                print("{}目录下无图片，请检查".format(os.path.join(root, sub_dir)))
                """
                这里的 0 表示程序因为一个预期的条件（数据集目录不存在）而退出，而不是因为程序本身的错误；
                如果需要让调用知道程序因为错误而异常退出，可以使用非零值，例如 sys.exit(1)。
                """
                sys.exit(0)

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
            print("已在 {} 创建分割数据集".format(split_dir))
