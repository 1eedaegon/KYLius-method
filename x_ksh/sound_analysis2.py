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
from sklearn.preprocessing import normalize
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
path = '/Users/kimseunghyuck/desktop/audio_train/'
files=os.listdir(path)

#show one sample file
filename = files[0]
y, sr = sf.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc.shape    #20,161

#show second sample file
filename = files[1]
y, sr = sf.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc.shape    #20,109

#show second sample file
filename = files[3]
y, sr = sf.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc.shape    #20,143

#show graph
plt.figure(figsize=(mfcc.shape[1]/15, 5))
plt.plot(mfcc)

#scaling for each frequency
max_mfcc=np.max(mfcc, axis=1)
mins, maxs=np.min(max_mfcc), np.max(max_mfcc)
scaled_mfcc=(max_mfcc-mins)/(maxs-mins)
scaled_mfcc

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

def data2array(file):
    dic = {}
    i=0
    for filename in file:
        y, sr = sf.read(path+filename, dtype='float32')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        max_mfcc=np.max(mfcc, axis=1)
        mins, maxs=np.min(max_mfcc), np.max(max_mfcc)
        scaled_mfcc=(max_mfcc-mins)/(maxs-mins)
        dic[i] = scaled_mfcc
        i+=1
    array=np.array(list(dic.values()))
    return(array)

trainData=data2array(trainfile)
testData=data2array(testfile)
print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 20) (2842, 20) (6631,) (2842,)

#how many kinds of label?
len(np.unique(trainLabel))   #41
len(np.unique(testLabel))    #41

#
idx = np.unique(trainLabel)
trainLabel=list(trainLabel)
trainLabel=pd.Series(trainLabel)

for i in range(len(idx)):
    for j in trainLabel:
        if idx[i]==j:
            trainLabel.loc['Oboe']
#여기서 실패
            
def one_hot(x):
    lst=[]
    for i in x:
        lst.append(np.repeat(0,41))
        lst[-1][i]+=1
    return(lst)
trainLabel=one_hot(trainLabel)
validateLabel=one_hot(testLabel)
#

#다시 돌릴때는 여기부터 
tf.reset_default_graph()     #그래프 초기화

# hyper parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
steps_for_validate = 5

#placeholders
X = tf.placeholder(tf.float32, [None, 20], name="X") 
X1 = tf.reshape(X, [-1, 10, 10])
Y = tf.placeholder(tf.string, [None, 1], name="Y")
keep_prob = tf.placeholder(tf.float32)

#cells
cells = tf.nn.rnn_cell.MultiRNNCell([DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=256),
                                                    output_keep_prob=keep_prob) for _ in range(3)])
h0 = cells.zero_state(batch_size, dtype=tf.float32)
output, hs = tf.nn.dynamic_rnn(cells, inputs=X1, initial_state=h0)
L1 = output[:, -1, :]

W2 = tf.Variable(tf.random_normal([256, 41]))
b2 = tf.Variable(tf.random_normal([41]))
hypothesis = tf.matmul(L1, W2) + b2

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
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: .5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: testData, Y: testLabel.reshape(-1, 1), keep_prob: 1}))
        save_path = saver.save(sess, '/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/승혁/optx/optx')
print('Finished!')