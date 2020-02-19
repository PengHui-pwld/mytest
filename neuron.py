"""
Life is short, you need python
_@Author_   :penghui
_@Time_     :2020/2/19&10:35   
"""
import _pickle as cPickle
import numpy as np
import os

CIFRA_DIR="./cifar-10-batches-py"
print(os.listdir(CIFRA_DIR))

with open(os.path.join(CIFRA_DIR,"data_batch_1"),'rb')as f:#读取data_batch_1里面的内容
    data= cPickle.load(f,encoding='bytes')
    print(type(data))#<class 'dict'>
    print(data.keys())#四个类 batch_label(批处理标签)filename(文件名) labels(标签) data(数据)

    print(type(data[b'data']))# 'numpy.ndarray'矩阵
    print(type(data[b'labels']))
    print(type(data[b'batch_label']))
    print((type(data[b'filenames'])))

    image_arr=data[b'data'][100]
    image_arr=image_arr.reshape((3.,32,32))#3表示3个维度，32 32分别表示图片大小32*32
    image_arr=image_arr.transpose((1,2,0))#图片通道显示顺序为 32 32 3 所以需要交换通道

    print(data[b'data'].shape)#(10000, 3072)10000*3072的矩阵，10000表示batch_1里面有10000张图片
                              #3072 图片展开，3个维度合并在一起，图片大小是32*32=1024 1024*3=3072
                              #3表示颜色的3通道，
    print(data[b'data'][0:2])#[ 59  43  50 ... 140  84  72]在0-255之间，像素点
    print(data[b'labels'][0:2])#[6, 9]表示第7和第10类 数据集有10个类
    print(data[b'batch_label'])
    print(data[b'filenames'][0:2])#RR-GG-BB=3072

