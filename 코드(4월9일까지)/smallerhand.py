# 꼬인 것 같음 ^^; 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:23:47 2018 

@author: kimseunghyuck(smallerhand)
"""
## 개선할 사항: layer수 늘리기, weight 초기값 xavier(오류나서 못했음), activation function 은 maxout을 쓰자. 
## dropout(앙상블). adam optimizer
import tensorflow as tf
import random
import pandas as pd
import numpy as np
tf.set_random_seed(777)  # reproducibility
train = pd.read_csv('desktop/python/train.csv')
test = pd.read_csv('desktop/python/test.csv')
trainData = train.values[:,1:]
#one hot function(n=10)
def one_hot(x):
    lst=[]
    for i in x:
        lst.append([0,0,0,0,0,0,0,0,0,0])
        lst[-1][i]+=1
    return(lst)
trainLabel=one_hot(train.values[:,0])
# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 256]))
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.Variable(tf.random_normal([256, 10]))
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L3, W4) + b4

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
saver = tf.train.Saver()
saver.save(sess, 'desktop/git/KYLius-method/save')

#0.941~0.9461보다 높길.
loaded_Graph = tf.Graph()
with tf.Session(graph=loaded_Graph) as sess:
    loader = tf.train.import_meta_graph('desktop/git/KYLius-method/save.meta')
    loader.restore(sess, 'desktop/git/KYLius-method/save.meta')    
    # get tensors
    loaded_x = loaded_Graph.get_tensor_by_name('input:0')
    loaded_y = loaded_Graph.get_tensor_by_name('label:0')
    loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')
    prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: testData})

import csv
header = ['ImageID','Label']
with open('desktop/git/KYLius-method/output_smallerhand.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter = ',')
    writer.writerow(header)
    for i, p in enumerate(prob):
        writer.writerow([str(i+1), str(p)])

    
    
