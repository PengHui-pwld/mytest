"""
Life is short, you need python
_@Author_   :penghui
_@Time_     :2020/2/19&21:27   
"""
#多个神经元实现多分类 w=10 有10个神经元
import tensorflow as tf
import os
import _pickle as cPickle
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 屏蔽GPU警告

CIFRA_DIR="./cifar-10-batches-py"
print(os.listdir(CIFRA_DIR))

def load_data(filename):
    """从数据文件读取数据"""
    with open(filename,'rb') as f:
        data=cPickle.load(f,encoding='bytes')
        return data[b'data'],data[b'labels']

#tensorflow.Dataset
class CifarData:
    def __init__(self,filenames,need_shuffle): #shuuffle表示将数据离散化 训练集需要，测试集不需要
        all_data=[]
        all_labels=[]
        for filenames in filenames:
            data,labels = load_data(filenames)
            all_data.append(data)
            all_labels.append(labels)
        self._data=np.vstack(all_data)#vstack 纵向上合并
        self._data=self._data/127.5-1#缩放(0-255)/127.5 在(0-2)之间，再-1 缩放到(-1 1)
        """在不归一化时结果在50%左右，在0 1之间会导致结果偏向一方 sigmod在偏向一方时，梯度会消失"""
        self._labels=np.hstack(all_labels)#横向合并
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples=self._data.shape[0]
        self._need_shuffle=need_shuffle
        self._indicator=0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p=np.random.permutation(self._num_examples)#0到_num_examples混排
        self._data=self._data[p]
        self._labels=self._labels[p]

    def next_batch(self,batch_size):
        """返回batch_size个样本"""
        end_indicator=self._indicator+batch_size
        if end_indicator>self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator=0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")

        if end_indicator>self._num_examples:
            raise Exception("batch size is larger than zhe examples")

        batch_data=self._data[self._indicator:end_indicator]
        batch_labels=self._labels[self._indicator:end_indicator]
        self._indicator=end_indicator
        return batch_data,batch_labels

train_filenames=[os.path.join(CIFRA_DIR,'data_batch_%d'%i)for i in range(1,6)]
test_filenames=[os.path.join(CIFRA_DIR,'test_batch')]

train_data=CifarData(train_filenames,True)
test_data=CifarData(test_filenames,False)
batch_data,batch_labels=train_data.next_batch(10)
print(batch_data)
print(batch_labels)
""""""
x=tf.placeholder(tf.float32,[None,3072])#tensorflow 先搭建图，然后再图里面填充数据，
                                        #placeholder相当于占位符，相当于变量，用来构建图
                                        #3072 这里按一维处理 表示3072个维度
"""[None,3072]"""

y=tf.placeholder(tf.int64,[None])  #None 表示第一位，样本数是不确定的，与bach_size对应
                                    #表示一个维度
                                    #get_variable获取变量函数
"""[None]"""

w=tf.get_variable('w',[x.get_shape()[-1],10],#[-1]与切片类似，对应x[]最好一个值(3072),1
                  initializer=tf.random_normal_initializer(0,1))
                                            #1表示输出，因为是二分类分离器，所有只有一个输出
                                            #initializer 初始化w 采用的方法tf.random_normal_initializer()
                                            #表示正态分布(0,1),标准正态分布
"""w: 3072*1 (3072,10)"""

b=tf.get_variable('b',[10],initializer=tf.constant_initializer(0.0))
                                            #b初始化为常量0
"""b:(10,)"""

y_=tf.matmul(x,w)+b #matmul矩阵乘法
"""[None,3072]*[3072,10]=[None,10]"""

""" 平方差损失函数 acc=40%
#course:1+e^x
#api:e^x/sum(e^x)
#一个概率分布有10个值，[[0.01,0.9.....0.03],[.....]]
p_y=tf.nn.softmax(y_)
#5->[0,0,0,0,0,1,0,0,0,0,0]
y_one_hot=tf.one_hot(y,10,dtype=tf.float32)
loss=tf.reduce_mean(tf.square(y_one_hot - p_y))#reduce_mean求均值，square()求平方
"""

loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
#y_ -> softmax
#y ->one_hot
#loss=ylogy_ 交叉熵  acc=33%


"""[1,0,1,0,1,0,1,0,1,1]"""
predict=tf.argmax(y_,1)
"""[1,1,0,1,0]"""
correct_predicton=tf.equal(predict,y)#预测正确的预测值
accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float64))

with tf.name_scope('train_op'):
    train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)#梯度下降方法

"""初始化函数"""
init=tf.global_variables_initializer()
batch_size=20
train_steps=100000
test_steps=100

#tensorflow 开启session 表示开始执行，session(绘画)
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels=train_data.next_batch(batch_size)
        loss_val,acc_val,_ =sess.run([loss,accuracy,train_op],feed_dict={x:batch_data,y:batch_labels})
         #有train_op 表示这次计算有训练，没有train_op表示测试，没有训练
         #feed_dict是要填充的数据，x,y cifar的图片和labels数据
        if (i+1)%500==0:
            print('[Train] Step: %d,loss:%4.5f,acc:%4.5f'%(i+1,loss_val,acc_val))

        if (i+1)%5000==0:
            test_data=CifarData(test_filenames,False)
            all_test_acc_val=[]
            for j in range(test_steps):
                test_batch_data,test_batch_labels=test_data.next_batch(batch_size)
                test_acc_val=sess.run([accuracy],feed_dict={x: test_batch_data,y:test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc=np.mean(all_test_acc_val)
            print('[Test] Step: %d,acc:%4.5f' %(i+1,test_acc))