#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 05:53:22 2018

@author: itwill03
"""


import numpy as np
import os
import glob
import tensorflow as tf
import pandas as pd

tf.set_random_seed(777) 

train_info = pd.read_csv("C:/data/sound/train.csv",delimiter=',')
train_data = np.genfromtxt("C:/data/sound/feature_train.csv", delimiter=',')


#label set
labels = train_info['label']
df_label = pd.DataFrame(labels)
l = train_info['label'].unique()

for i in range(len(l)):
    df_label[df_label==l[i]] = i
    
#train data set    
train_data = pd.DataFrame(train_data)
train_data['label']=df_label
train_data = train_data.astype(np.float32)

#np.savetxt("c:/data/sound/train_data.csv",train_data, delimiter=",")

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train_data, test_size = 0.3)
trainData = train_set.values[:,0:193]  
trainLabel = train_set.values[:,-1]
validateData = validate_set.values[:,0:193]
validataLabel = validate_set.values[:,-1]

# 텐서플로우 모델 생성
n_dim = 193
n_classes = 41
training_epochs = 10
learning_rate = 0.001
batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name='p_keep_conv')
p_keep_hidden = tf.placeholder(tf.float32, name='p_keep_hidden')


c1 = tf.layers.conv2d(tf.reshape(X, [-1, 1, n_dim, 1]), 32, kernel_size=[1, 5], strides=(1, 1), padding='same', 
                      activation=tf.nn.elu, name="c1")  
p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[1, 2], strides=2) 
p1 = tf.nn.dropout(p1, p_keep_conv)

# shape=(?, 1, 96, 32)
c2 = tf.layers.conv2d(p1, 64, kernel_size=[1,5], strides=(1, 1), padding='same', 
                      activation=tf.nn.elu, name="c2")
p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[1, 2], strides=2) #shape = [?, 1, 48, 100]
p2 = tf.nn.dropout(p2, p_keep_conv)

# shape=(?, 1, 48, 64)
c3 = tf.layers.conv2d(p2, 128, kernel_size=[1,5], strides=(1, 1), padding='same', 
                      activation=tf.nn.elu, name="c3")
p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[1, 2], strides=2) #shape = [?, 1, 24, 200]
p3 = tf.nn.dropout(p3, p_keep_conv)

L4_flat = tf.reshape(p3, shape=[-1, 1*24*128]) 
W1 = tf.get_variable("W1", shape=[1*24*128, 624], initializer=tf.contrib.layers.xavier_initializer())
L5 = tf.nn.relu(tf.matmul(L4_flat, W1))
L5 = tf.nn.dropout(L5, p_keep_hidden)

W2 = tf.get_variable("W2", shape=[624,41],initializer=tf.contrib.layers.xavier_initializer())
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
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validataLabel.reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))       
print('Finished!')

tf.reset_default_graph() 
"""
#1
training_epochs = 500
learning_rate = 0.001
sigmoid
conv2d kenel size = 1,2
flat = 1000
adam
cost = 1.203582104
ccuracy: 0.476073

#2  'sigmoid -> relu'

training_epochs = 500
learning_rate = 0.001
relu
conv2d kenel size = 1,2
flat = 1000
adam
cost = 0.021049510
Accuracy: 0.626319

#3 'kanel size = 1,2 -> 1,5'

training_epochs = 500
learning_rate = 0.001
relu
conv2d kenel size = 1,5
flat = 1000
adam
cost = 0.026283483
Accuracy: 0.626671

#4 relu -> elu
Accuracy: 0.64145

#5 conv2d layer +1, drop out
cost = 0.300675828
Accuracy: 0.632301
"""
