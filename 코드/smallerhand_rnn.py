#mlp
#테스트 데이터에 라벨이 없어서 정확도를 (캐글에 올리기 전엔) 확인할 수 없기 때문에 training 데이터를 7:3으로 나눠서 검증했습니다.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:23:47 2018 
@author: kimseunghyuck(smallerhand)
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
import pandas as pd
import numpy as np
train = pd.read_csv('desktop/python/train.csv')
test = pd.read_csv('desktop/python/test.csv')

#훈련세트, validation세트 나누기
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777)  # reproducibility
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]

# parameters
learning_rate = 0.002
training_epochs = 10
batch_size = 100
steps_for_print = 5
steps_for_validate = 10
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28])
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10])

# layers
cells = tf.nn.rnn_cell.MultiRNNCell([DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=256),
                                                    output_keep_prob=keep_prob) for _ in range(3)])
h0 = cells.zero_state(batch_size, dtype=tf.float32)
output, hs = tf.nn.dynamic_rnn(cells, inputs=X_img, initial_state=h0)
L1 = output[:, -1, :]

W2 = tf.Variable(tf.random_normal([256, 10]))
b2 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L1, W2) + b2

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize 
sess = tf.Session()
sess.run(tf.global_variables_initializer())  #글로벌 변수 초기화

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: .5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict) 
        avg_cost += c / total_batch         
    if epoch % steps_for_print == steps_for_print - 1:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate - 1:
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1), keep_prob: 1}))       
print('Finished!')

"""
lstm cell 넣었음. 돌아가는 것 같긴 한데 너무 부하가 큰 것 같아서 겁나서 멈춤. GPU로 돌려보겠음.
돌아가는지만 보려고 시험삼아 total_batch를 줄여서 돌려봤는데 돌아가고 출력되긴함.
"""
