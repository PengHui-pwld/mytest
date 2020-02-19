"""
Life is short, you need python
_@Author_   :penghui
_@Time_     :2020/2/19&21:27   
"""
import tensorflow as tf
import os
import _pickle as cPickle
import numpy as np

CIFRA_DIR="./cifar-10-batches-py"
print(os.listdir(CIFRA_DIR))

class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data=[]
        all_labels=[]
        for filenames in filenames:
            #data,labels=load(filenames)
            """明天继续"""

def load_data(filename):
    """从数据文件读取数据"""
    with open(filename,'rb') as f:
        data=cPickle.load(f,encoding='bytes')
        return data['data'],data['labels']

x=tf.placeholder(tf.float32,[None,3072])#tensorflow 先搭建图，然后再图里面填充数据，
                                        #placeholder相当于占位符，相当于变量，用来构建图
                                        #3072 这里按一维处理 表示3072个维度
"""[None,3072]"""

y=tf.placeholder(tf.int64,[None])  #None 表示第一位，样本数是不确定的，与bach_size对应
                                    #表示一个维度
                                    #get_variable获取变量函数
"""[None]"""

w=tf.get_variable('w',[x.get_shape()[-1],1],#[-1]与切片类似，对应x[]最好一个值(3072),1
                  initializer=tf.random_normal_initializer(0,1))
                                            #1表示输出，因为是二分类分离器，所有只有一个输出
                                            #initializer 初始化w 采用的方法tf.random_normal_initializer()
                                            #表示正态分布(0,1),标准正态分布
"""w: 3072*1 (3072,1)"""

b=tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))
                                            #b初始化为常量0
"""b:(1,)"""

y_=tf.matmul(x,w)+b #matmul矩阵乘法
"""[None,3072]*[3072,1]=[None,1]"""

p_y_1=tf.nn.sigmoid(y_)#预测值
"""[None,1]"""

y_reshaped=tf.reshape(y,(-1,1))#真实值
"""[None,1]"""

y_reshaped_float=tf.cast(y_reshaped,tf.float32)#类型一致,最后的真实值

loss=tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))#reduce_mean求均值，square()求平方

"""bool"""
predict=p_y_1>0.5#预测值 true false
"""[1,1,0,1,0]"""
correct_predicton=tf.equal(tf.cast(predict,tf.int64),y_reshaped)#预测正确的预测值
accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float64))

with tf.name_scope('train_op'):
    train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)#梯度下降方法

init=tf.global_variable_initializer()
with tf.Session() as sess:
    sess.run([loss,accuracy,train_op],feed_dict={x:,y:})