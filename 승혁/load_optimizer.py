#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 01:33:49 2018

@author: kimseunghyuck
"""

import tensorflow as tf
import numpy as np
import pandas as pd
#path="/home/paperspace/Downloads/"
path2="desktop/git/daegon/KYLius-method/승혁/"
#데이터 가져오기
train = pd.read_csv(path2+"train.csv")
#train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기(여기서는 validate만 필요)
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
#trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
#trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]

"""
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])          # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10])

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
#optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1)

print(validateData.shape)
#train 한 옵티마이저와서 맞추기
with tf.Session() as sess:
    saver=tf.train.import_meta_graph(path+"optimizer/opt.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint(path+"optimizer/"))
    print("Model restored.")
    print(sess.run(tf.argmax(logits,1), feed_dict={
            X: validateData}))  
"""
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.import_meta_graph(path2+"optimizer/opt.ckpt.meta")
path2="desktop/git/daegon/KYLius-method/승혁/"
saver.restore(sess, tf.train.latest_checkpoint(path2+"optimizer/"))
print("Model restored.")
path2="desktop/git/daegon/KYLius-method/승혁/"
# test the data
print('Learning started. It takes sometime.')
pred = sess.run(tf.argmax(logits, 1), feed_dict={X: validateData})
print(pred)
print('Finished!')