#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 05:53:22 2018
@author: itwill03
"""

# 전처리 코드 : mel_extract_128.py


import numpy as np
import os
import glob
import tensorflow as tf
import pandas as pd

tf.set_random_seed(777) 

train_info = pd.read_csv("/home/itwill03/sound/train.csv",delimiter=',')
train_data = np.genfromtxt("/home/itwill03/sound/mel_train2.csv", delimiter=',')
#train_info = pd.read_csv("C:\data\sound/train.csv",delimiter=',')
#train_data = np.genfromtxt("C:\data\sound\mel_train2.csv", delimiter=',')


#labels to number : 라벨은 0~41의 숫자로 전환
def labels2Num(labels):
    df_label = pd.DataFrame(labels)
    l = train_info['label'].unique()
    for i in range(len(l)):
        df_label[df_label==l[i]] = i
    return df_label

labels = train_info['label']
df_label = labels2Num(labels)    
    
#train data set    
train_data = pd.DataFrame(train_data)
train_data['label']=df_label
train_data = train_data.astype(np.float32)

#np.savetxt("c:/data/sound/train_data.csv",train_data, delimiter=",")

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train_data, test_size = 0.3)
trainData = train_set.values[:,0:16384]  
trainLabel = train_set.values[:,-1]
validateData = validate_set.values[:,0:16384]
validataLabel = validate_set.values[:,-1]

# graph reset
tf.reset_default_graph()

# 텐서플로우 모델 생성
n_dim = 16384
n_classes = 41
training_epochs = 700
learning_rate = 0.0004
batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name='p_keep_conv')
p_keep_hidden = tf.placeholder(tf.float32, name='p_keep_hidden')


#convolution layer1
    # img shape = (?, 128, 128, 1)
c1 = tf.layers.conv2d(tf.reshape(X, [-1, 128, 128, 1]), 32, kernel_size=[5, 3], strides=(1, 1), 
                      padding='same', activation=tf.nn.relu, name="c1")
n1 = tf.layers.batch_normalization(c1)  
p1 = tf.layers.max_pooling2d(inputs=n1, pool_size=[4, 4], strides=4) 
p1 = tf.nn.dropout(p1, p_keep_conv)

#convolution layer2
    # img shape = (?, 32, 32, 32)
c2 = tf.layers.conv2d(tf.reshape(p1, [-1, 32, 32, 32]), 64, kernel_size=[5, 3], strides=(1, 1), 
                      padding='same', activation=tf.nn.relu, name="c2")
n2 = tf.layers.batch_normalization(c2) 
p2 = tf.layers.max_pooling2d(inputs=n2, pool_size=[3, 3], strides=2) #shape = [?, 1, 48, 100]
p2 = tf.nn.dropout(p2, p_keep_conv)

#convolution layer3
    # img shape = (?, 15, 15, 64)
c3 = tf.layers.conv2d(tf.reshape(p2, [-1, 15, 15, 64]), 128, kernel_size=[5, 3], strides=(1, 1), 
                      padding='same', activation=tf.nn.relu, name="c3")
n3 = tf.layers.batch_normalization(c3) 
p3 = tf.layers.max_pooling2d(inputs=n3, pool_size=[3, 3], strides=2) #shape = [?, 1, 24, 200]
p3 = tf.nn.dropout(p3, p_keep_conv)

#flating layer, hidenlayer  
L4_flat = tf.reshape(p3, shape=[-1, 7*7*128]) 
W1 = tf.get_variable("W1", shape=[7*7*128, 640], initializer=tf.contrib.layers.xavier_initializer())
L5 = tf.nn.relu(tf.matmul(L4_flat, W1))
n5 = tf.layers.batch_normalization(L5) 
L5 = tf.nn.dropout(n5, p_keep_hidden)

W2 = tf.get_variable("W2", shape=[640,41],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([41]))
logits = tf.matmul(L5, W2) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1, name="pred")

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, p_keep_conv: .7, p_keep_hidden: .5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        x=np.random.choice(validataLabel.shape[0], 500, replace=False)
        print('Accuracy:', sess.run(accuracy, feed_dict={
            X: validateData[x], Y: validataLabel[x].reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))
        #save_path = saver.save(sess, '/home/paperspace/Downloads/optx/optx')
print('Finished!')

"""
#1.
training_epochs = 700
learning_rate = 0.0004
activation = elu
conv = kernel_size=[3, 3], strides=(1, 1)
       kernel_size=[3, 3], strides=(1, 1)
       kernel_size=[3, 3], strides=(1, 1)
pool = pool_size=[2, 2], strides=2
        pool_size=[2, 2], strides=2
        pool_size=[2, 2], strides=2
flat = 16*16*128 -> 64
p_keep_conv: .7, p_keep_hidden: .5
cost : 0.06~0.08에서 계속 떨어지는중
Accuracy: 59~63% 
전처리시 정규화 고려.. librosa.util.normalize
윈도우 14*2 고려..
러닝레이트 조절

#2.
training_epochs = 700
learning_rate = 0.0001
activation = relu
conv = kernel_size=[5, 3], strides=(1, 1)
        kernel_size=[5, 3], strides=(1, 1)
        kernel_size=[5, 3], strides=(1, 1)
pool = pool_size=[4, 4], strides=4
        pool_size=[3, 3], strides=2
        pool_size=[3, 3], strides=2
flat = 7*7*128, 640
p_keep_conv: .7, p_keep_hidden: .5
cost = 0.122217526
Accuracy: 60~67% 

#3.
training_epochs = 700
learning_rate = 0.0004
activation = relu
conv = kernel_size=[5, 3], strides=(1, 1)
	kernel_size=[5, 3], strides=(1, 1)
	kernel_size=[5, 3], strides=(1, 1)
pool = pool_size=[12, 4], strides=4
	pool_size=[12, 4], strides=2
	pool_size=[2, 2], strides=2
flat = 5*7*128, 420
p_keep_conv: .8, p_keep_hidden: .7
Accuracy: 57~64%



"""
