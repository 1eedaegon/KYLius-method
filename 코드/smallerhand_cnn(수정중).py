#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:23:47 2018 
@author: kimseunghyuck(smallerhand)
"""
import tensorflow as tf
tf.reset_default_graph()     #그래프 초기화
import pandas as pd
import numpy as np
train = pd.read_csv('desktop/python/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]

#one hot function(n=10)
def one_hot(x):
    lst=[]
    for i in x:
        lst.append([0,0,0,0,0,0,0,0,0,0])
        lst[-1][i]+=1
    return(lst)
trainLabel=one_hot(train_set.values[:,0])
validateLabel=one_hot(validate_set.values[:,0])

# parameters
learning_rate = 0.002
training_epochs = 40
batch_size = 100
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

# Layer 1
W1 = tf.Variable(tf.random_normal([4,4,1,32], stddev=.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],
                    strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.Variable(tf.random_normal([4,4,32,64], stddev=.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],
                    strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2_flat = tf.reshape(L2, [-1, 64 * 5 * 5])

# Layer 3 FC 5x5x64 inputs -> 625 outputs
W3 = tf.get_variable("W3", shape=[64 * 5 * 5, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([625]))
L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# Layer 4 Final FC 625 inputs -> 10 outputs
W4 = tf.get_variable("W4", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3, W4) + b4

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

# Test model and check accuracy(validation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: validateData, Y: validateLabel}))
    
