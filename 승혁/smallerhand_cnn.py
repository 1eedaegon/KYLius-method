#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:27:00 2018
    
@author: kimseunghyuck
"""
import tensorflow as tf
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777) 
import pandas as pd
import numpy as np
train = pd.read_csv('desktop/python/train.csv')
#train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]

# hyper parameters

learning_rate = 0.0001
training_epochs = 5
#gpu로 돌릴때는 100까지 돌림

batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])          # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')

L1 = tf.nn.elu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 ImgIn shape=(?, 14, 14, 10)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# add L1_2
W1_2 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01))
L1_2 = tf.nn.conv2d(X_img, W1_2, strides=[1, 1, 1, 1], padding='SAME')
L1_2 = tf.nn.elu(L1_2)
L1_2 = tf.nn.max_pool(L1_2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1_2 = tf.nn.dropout(L1_2, keep_prob=keep_prob)

# L2 ImgIn shape=(?, 14, 14, 10)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# L3
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

# Final FC 7x7x64 inputs -> 10 outputs
W4_1 = tf.get_variable("W4_1", shape=[4 * 4 * 128, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat, W4_1) + b

# add L2_2
W2_2 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))
L2_2 = tf.nn.conv2d(L1_2, W2_2, strides=[1, 1, 1, 1], padding='SAME')
L2_2 = tf.nn.elu(L2_2)
L2_2 = tf.nn.max_pool(L2_2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_sum = L2+L2_2
L2_2 = tf.nn.dropout(L2_sum, keep_prob=keep_prob)

# L3
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2_2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=0.9)
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

# Final FC 7x7x64 inputs -> 10 outputs
W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 10],
                     initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat, W4) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y_onehot))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: .7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1), keep_prob: 1}))       
print('Finished!')


"""
learning_rate = 0.001
training_epochs = 15
batch_size = 100
1. 기존꺼에서 병렬 레이어 제거
98.34%(epoch 5)
2. 상욱이가 올린 김성훈교수 코드 인수 넣음.
98.77%(epoch 5)
98.94%(epoch 20)
99.10%(epoch 20)

3. epoch 늘림(100~200)
99.1%~99.2%
4. relu -> elu 로 바꿈.
Adam -> RMSPropOptimizer
99.3%대(epoch 100~300)

"""
