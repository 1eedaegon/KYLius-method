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
training_epochs = 40
batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([256, 256]))
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([256, 10]))
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L3, W4) + b4

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
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 1}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict) 
        avg_cost += c / total_batch     
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate ==0:
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1), keep_prob: 1}))       
print('Finished!')

#최종 정확도 측정
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
        X: validateData, Y: validateLabel.reshape(-1, 1), keep_prob: 1}))  
"""
4개 layer(hidden 2개), learning rate=0.002, training epoch =40, batch size =100
각 레이어의 인풋과 아웃풋은 784->256->256->256->10
1. 기본
정확도 95.90%, 95.69%, 95.04%, 95.17%, 95.54%, 95.65%, 95.41%, 95.60%
2. dropout 이용
keep_prob=0.5 : 정확도 13.07%, keep_prob=0.7: 정확도 35.38%
일단 droupout 이용안하고 다른 걸 바꿔서 정확도 높여봐야 하겠다.


"""
