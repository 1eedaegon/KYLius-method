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
trainData.shape  #(29400, 784)
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
trainData = trainData.reshape(trainData.shape[0], 28,28)
trainData.shape  #(29400, 28, 28)
validateData = validateData.reshape(validateData.shape[0], 28,28)

# parameters
learning_rate = 0.002
training_epochs = 40
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=10, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, 10])
outputs = tf.contrib.layers.fully_connected(X_for_fc, 10, activation_fn=None)

# reshape out for sequence_loss
hypothesis = tf.reshape(outputs, [-1, 10])

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
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
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

# Test model and check accuracy(validation)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: validateData, Y: validateLabel}))
    
