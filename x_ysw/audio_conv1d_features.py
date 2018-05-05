# -*- coding: utf-8 -*-
"""
Created on Wed May  2 19:21:50 2018

@author: stu
"""

#전처리 code : extract_features.py

import numpy as np
import os
import glob
import tensorflow as tf
import pandas as pd

tf.set_random_seed(777) 

train_info = pd.read_csv("C:/data/sound/train.csv",delimiter=',')
train_data = np.genfromtxt("C:/data/sound/feature_train.csv", delimiter=',')

#label set : 라벨 값을 0~41로 바꿈

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

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train_data, test_size = 0.3)
trainData = train_set.values[:,0:193]  
trainLabel = train_set.values[:,-1]
validateData = validate_set.values[:,0:193]
validataLabel = validate_set.values[:,-1]


# 텐서플로우 모델 생성
tf.reset_default_graph()

n_dim = 193
n_classes = 41
training_epochs = 700
learning_rate = 0.001
batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, n_dim])
X_1d = tf.reshape(X, [-1,193,1])
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name='p_keep_conv')
p_keep_hidden = tf.placeholder(tf.float32, name='p_keep_hidden')
193*1.5
#convolution layer 1
c1 = tf.layers.conv1d(X_1d, 386,kernel_size=2, strides=1, padding='Same',
                     activation=tf.nn.relu, name='c1')
n1 = tf.layers.batch_normalization(c1)
p1 = tf.layers.max_pooling1d(n1, pool_size=2, strides=2, padding='Same')
p1 = tf.nn.dropout(p1, p_keep_conv)
386*1.5
#shape=(?, 97, 386)
c2 = tf.layers.conv1d(p1, 579,kernel_size=2, strides=1, padding='Same',
                     activation=tf.nn.relu, name='c2')
n2 = tf.layers.batch_normalization(c2)
p2 = tf.layers.max_pooling1d(n2, pool_size=2, strides=2, padding='Same')
p2 = tf.nn.dropout(p2, p_keep_conv)

#shape=(?, 49, 579)
c3 = tf.layers.conv1d(p2, 579,kernel_size=1, strides=1, padding='Same',
                     activation=tf.nn.relu, name='c3')
n3 = tf.layers.batch_normalization(c3)
p3 = tf.layers.max_pooling1d(n3, pool_size=2, strides=2, padding='Same')
p3 = tf.nn.dropout(p3, p_keep_conv)

#shape=(?, 25, 579)
L4_flat = tf.reshape(p3, [-1,25*579])

W4 = tf.get_variable("W4", shape=[25*579, 624], initializer=tf.contrib.layers.xavier_initializer())
L5 = tf.nn.relu(tf.matmul(L4_flat, W4))
n5 = tf.layers.batch_normalization(L5)
L5 = tf.nn.dropout(n5, p_keep_hidden)

W5 = tf.get_variable("W5", shape=[624,41], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([41]))
logits = tf.matmul(L5, W5) + b

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


