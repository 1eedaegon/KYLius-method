#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:53:10 2018

@author: kimseunghyuck
"""

#import important modules
import numpy as np
import tensorflow as tf
tf.set_random_seed(777) 

trainData = np.genfromtxt('/home/paperspace/Downloads/trainData5.csv', delimiter=',')
trainData = trainData.reshape(-1, 20, 430)
testData = np.genfromtxt('/home/paperspace/Downloads/testData5.csv', delimiter=',')
testData = testData.reshape(-1, 20, 430)
trainLabel = np.genfromtxt('/home/paperspace/Downloads/trainLabel5.csv', delimiter=',')
testLabel = np.genfromtxt('/home/paperspace/Downloads/testLabel5.csv', delimiter=',')

print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 20, 430) (2842, 20, 430) (6631,) (2842,)

#다시 돌릴때는 여기부터 
tf.reset_default_graph()     #그래프 초기화

# hyper parameters
learning_rate = 0.0002
training_epochs = 700
batch_size = 50
steps_for_validate = 5

#placeholder
X = tf.placeholder(tf.float32, [None, 20, 430], name="X")
X_sound = tf.reshape(X, [-1, 20, 430, 1])          # 20*100*1 (frequency, time, amplitude)
Y = tf.placeholder(tf.int32, [None, 1], name="Y")
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

# L1 SoundIn shape=(?, 20, 430, 1)
W1 = tf.get_variable("W1", shape=[2, 43, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_sound, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.layers.batch_normalization(L1)
L1 = tf.nn.elu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 43, 1],strides=[1, 2, 43, 1], padding='SAME') 
L1 = tf.nn.dropout(L1, p_keep_conv)

# L2 Input shape=(?,10,21,32)
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.layers.batch_normalization(L2)
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
L2 = tf.nn.dropout(L2, p_keep_conv)

# L3 Input shape=(?,3,12,64)
W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.layers.batch_normalization(L3)
L3 = tf.nn.elu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
L3 = tf.nn.dropout(L3, p_keep_conv)
L3_flat= tf.reshape(L3, shape=[-1, 3*3*128])

# Final FC 2*3*128 inputs -> 41 outputs
W4 = tf.get_variable("W4", shape=[3*3*128, 512],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.elu(tf.matmul(L3_flat, W4))
L4 = tf.nn.dropout(L4, p_keep_hidden)
W_o = tf.get_variable("W_o", shape=[512,41],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([41]))
logits = tf.matmul(L4, W_o) + b

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
        feed_dict = {X: batch_xs, Y: batch_ys, 
                     p_keep_conv: .8, p_keep_hidden: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        x=np.random.choice(testLabel.shape[0], 500, replace=False)
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: testData[x], Y: testLabel[x].reshape(-1, 1), 
                p_keep_conv: 1, p_keep_hidden: 1}))
        save_path = saver.save(sess, '/home/paperspace/Downloads/optx/optx')
print('Finished!')

#마지막에 정확도랑 코스트 그래프로 출력해주는 코드 넣기
#정규화/ stft
#label 1이랑 0이랑 나누기
"""
9) data5 (n_mfcc=20, length=430)
lr=0.0002, epoch = 200    
win : (2, 21), (2,4), (3,3)
max_pool : (2,21), (2,4), (3,3)
p_keep_conv, p_keep_hidden = 0.8, 0.7
accuracy : 63~70% 

10) 9와 같고
lr=0.0002, epoch = 700    
win : (2, 43), (3,3), (3,3)
max_pool : (2,43), (3,3), (3,3)
p_keep_conv, p_keep_hidden = 0.8, 0.7
accuracy : 66~74%

11) 9와 같고
lr=0.0002, epoch = 700
p_keep_conv, p_keep_hidden = 0.7, 0.5
accuracy : 67~72%
"""
