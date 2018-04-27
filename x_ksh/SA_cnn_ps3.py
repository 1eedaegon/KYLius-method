#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:53:10 2018

@author: kimseunghyuck
"""

import librosa
import soundfile as sf
import numpy as np

#import labels
import tensorflow as tf
tf.set_random_seed(777) 
import pandas as pd
#train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
train = pd.read_csv('/home/paperspace/Downloads/audio_train.csv')

#train/test, Data/Label split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.3)
trainfile = train_set.values[:,0]
testfile = test_set.values[:,0]
trainLabel = train_set.values[:,1]
testLabel = test_set.values[:,1]

#data load and extract mfcc (scaling indluded)
#path = '/Users/kimseunghyuck/desktop/audio_train/'
path = '/home/paperspace/Downloads/audio_train/'

def see_how_long(file):
    c=[]
    for filename in file:
        y, sr = sf.read(path+filename, dtype='float32')
        mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        abs_mfcc=np.abs(mfcc)
        #1025 X t 형태
        c.append(abs_mfcc.shape[1])
    return(c)
 
#n=see_how_long(trainfile)
#print(np.max(n), np.min(n))      #2584 28
#n2=see_how_long(testfile)
#print(np.max(n2), np.min(n2))    #2584 26

#zero padding to 3000
def data2array(file):
    #zero padding to file.shape[0] X 20 X 3000    
    n=file.shape[0]
    k=0
    array = np.repeat(0, n * 20 * 3000).reshape(n, 20, 3000)
    filename=file[0]
    for filename in file:
        y, sr = sf.read(path+filename, dtype='float32')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        abs_mfcc=np.abs(mfcc)
        for j in range(abs_mfcc.shape[1]):
            array[k, :, j]=abs_mfcc[:, j]
        k+=1
    return(array)

trainData=data2array(trainfile)
testData=data2array(testfile)

print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 20, 3000) (2842, 20, 3000) (6631,) (2842,)

"""
#csv다운로드 했는데 train, test 합쳐서 15GB라 안올리고 생략.
trainData2D=trainData.reshape(-1, 20*3000)
testData2D=testData.reshape(-1, 20*3000)
np.savetxt('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh/trainData.csv', 
           trainData2D, delimiter=",")
np.savetxt('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh/testData.csv', 
           testData2D, delimiter=",")
"""


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
learning_rate = 0.001
training_epochs = 500
batch_size = 100
steps_for_validate = 5

#placeholder
X = tf.placeholder(tf.float32, [None, 20, 3000], name="X")
X_sound = tf.reshape(X, [-1, 20, 3000, 1])          # 20*3000*1 (frequency, time, amplitude)
Y = tf.placeholder(tf.int32, [None, 1], name="Y")
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

# L1 SoundIn shape=(?, 20, 3000, 1) 3000=20*150
W1 = tf.get_variable("W1", shape=[2, 20, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_sound, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.elu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 20, 20, 1],strides=[1, 20, 20, 1], padding='SAME') # l1 shape=(?, 14, 14, 32)
L1 = tf.nn.dropout(L1, p_keep_conv)
L1_flat= tf.reshape(L1, shape=[-1, 150*32])

# Final FC 4x4x128 inputs -> 10 outputs
W4 = tf.get_variable("W4", shape=[150*32, 625],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.elu(tf.matmul(L1_flat, W4))
L4 = tf.nn.dropout(L4, p_keep_hidden)
W_o = tf.get_variable("W_o", shape=[625,41],initializer=tf.contrib.layers.xavier_initializer())
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
        feed_dict = {X: batch_xs, Y: batch_ys, p_keep_conv: .7, p_keep_hidden: .5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: testData[:100], Y: testLabel[:100].reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))
        save_path = saver.save(sess, '/home/paperspace/Downloads/optx/optx')
print('Finished!')

#테스트 할 때 샘플 추출 하는 방법 바꾸기
#list(array1, array2, ...) 이런식으로 구성해야겠음. 안그러면 너무 커짐.
#A=[np.array([1,2,3]), np.array([2,3,4,5,1])]
#아래처럼 구성해볼 생각.
"""
W1(20, 20)
for i in range(length):
    L[i]=tanh(matmul(X, W1))
max(L)

에폭 1000, lr 0.001, 정확도 32.6~32.9% 
다른 조건 같고 keep_prop=0.8, 정확도 36.9~37.9%
"""