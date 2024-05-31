# -*- coding: utf-8 -*-

###### 
######
###### chaonin 24.5.14 from 51CTO
######
######

###### 步骤1: 直接读取latin1格式文件并还原为图片
import gzip
import pickle
import pylab

# i={0,1,2}：i=0时为50000个数据的训练集；i=1时为10000个数据的验证集；i=2时为10000个数据的测试集
# 输出第j张图片
print('你要读第几个库的手写数字-->0-50000个训练集;1-10000个验证集;2-10000个测试集:')
i = int(input())
print('你要读取这个库中的第几张图片:')
j = int(input())

# 以二进制只读格式读取图片及索引文件
with gzip.open('mnist.pkl.gz', 'rb') as f:
    img = pickle.load(f, encoding='latin1')
img_x = img[i][0][j].reshape(28, 28)
img_id = img[i][1][j]

print('这个数字是：'+str(img_id))
pylab.imshow(img_x)
pylab.gray()
pylab.show()

###### 步骤2: 用mnist_loader封装的方法输出矩阵化后的手写数字
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
if i==0:
    print(training_data[j])
elif i==1:
    print(validation_data[j])
elif i==2:
    print(test_data[j])