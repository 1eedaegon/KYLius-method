#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:53:10 2018

@author: kimseunghyuck
"""

import librosa
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import os
path = '/Users/kimseunghyuck/desktop/audio_train/'
files=os.listdir(path)

#show one sample file
filename = files[0]
y, sr = sf.read(path+filename, dtype='float32')
#mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
stft=librosa.core.stft(y=y)
stft.shape  #1025, 161

#show second sample file
filename = files[1]
y, sr = sf.read(path+filename, dtype='float32')
#mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
stft=librosa.core.stft(y=y)
stft.shape  #1025, 109
#주파수 범위가 1025, 109는 시간
#패딩해서 max로 맟춘 다음에 cnn하면 되지 않을까.

#show graph
plt.figure(figsize=(15, 5))
plt.plot(stft)

#show abs graph
stft=np.abs(stft)
plt.figure(figsize=(15, 5))
plt.plot(stft)

#import labels
import tensorflow as tf
tf.set_random_seed(777) 
import pandas as pd
train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')

#train/test, Data/Label split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.3)
trainfile = train_set.values[:,0]
testfile = test_set.values[:,0]
trainLabel = train_set.values[:,1]
testLabel = test_set.values[:,1]

#data load and extract mfcc (scaling indluded)
path = '/Users/kimseunghyuck/desktop/audio_train/'

def see_how_long(file):
    c=[]
    for filename in file:
        y, sr = sf.read(path+filename, dtype='float32')
        stft=librosa.core.stft(y=y)
        abs_stft=np.abs(stft)
        #1025 X t 형태
        c.append(abs_stft.shape[1])
    return(c)
 
#n=see_how_long(trainfile)
#print(np.max(n), np.min(n))      #2584 28
#n2=see_how_long(testfile)
#print(np.max(n2), np.min(n2))    #2584 26
file=trainfile

#zero padding to 3000
def data2array(file):
    #zero padding to file.shape[0] X 1025 X 3000
    n=file.shape[0]
    k=0
    array = np.repeat(0, n * 1025 * 3000).reshape(n, 1025, 3000)

    for filename in file:
        y, sr = sf.read(path+filename, dtype='float32')
        stft=librosa.core.stft(y=y)
        abs_stft=np.abs(stft)
        for i in range(abs_stft.shape[0]):
            for j in range(abs_stft.shape[1]):
                array[k, i, j]=abs_stft[k, i, j]
        k+=1
    return(array)


import numpy as np
x=np.repeat(0, 6000*20*5000).reshape(6000,20,5000)

trainData=data2array(trainfile)
testData=data2array(testfile)

print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 20) (2842, 20) (6631,) (2842,)

#how many kinds of label?
print(len(np.unique(trainLabel)))   #41
print(len(np.unique(testLabel)))    #41

#label string -> integer(0~40)
idx = np.unique(trainLabel)

def Labeling(label):
    r=pd.Series(label)
    for i in range(len(idx)):
        r[r.values==idx[i]]=i
    return(r)

trainLabel=Labeling(trainLabel)
testLabel=Labeling(testLabel)
print(min(trainLabel), max(trainLabel), min(testLabel), max(testLabel))

#다시 돌릴때는 여기부터 
tf.reset_default_graph()     #그래프 초기화

# hyper parameters
learning_rate = 0.0001
training_epochs = 5000
batch_size = 100
steps_for_validate = 5

#placeholders
X = tf.placeholder(tf.float32, [None, 20], name="X") 
Y = tf.placeholder(tf.int32, [None, 1], name="Y")
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([20, 128]))
b1 = tf.Variable(tf.random_normal([128]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([128, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([256, 41]))
b4 = tf.Variable(tf.random_normal([41]))
hypothesis = tf.matmul(L2, W4) + b4

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: .8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: testData, Y: testLabel.reshape(-1, 1), keep_prob: 1}))
        save_path = saver.save(sess, '/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh/optx/optx')
print('Finished!')

"""
에폭 1000, lr 0.001, 정확도 32.6~32.9% 
다른 조건 같고 keep_prop=0.8, 정확도 36.9~37.9%
"""