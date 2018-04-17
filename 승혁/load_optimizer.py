#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 01:33:49 2018

@author: kimseunghyuck
"""

import tensorflow as tf
import numpy as np
import pandas as pd
path="/home/paperspace/Downloads/"

#데이터 가져오기
train = pd.read_csv(path+"train.csv")
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
"""

print(validateData.shape)
#train 한 옵티마이저와서 맞추기
with tf.Session() as sess:
    saver=tf.train.import_meta_graph(path+"opt.ckpt.meta")
    saver.restore(sess, path+"opt.ckpt")
    print("Model restored.")
    print(sess.run(tf.argmax(logits,1), feed_dict={
            X: validateData}))  
