# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:01:08 2018

@author: modes
"""

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777) 
import pandas as pd



#train = pd.read_csv('c:/python/train.csv')
train = pd.read_csv('/data/mnist/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]


"""
pixel 계산
trainData.shape #Out[137]: (29400, 784)
num,img=trainData.shape 
np.sqrt(img) #pixel 계산
#Out[141]: 28.0  
"""

img = trainData
label = trainLabel

#단일 image pixel 계산 함수
def how_pix(data):
    num, img = data.shape
    res = int(np.sqrt(img))
    return(res,res)

pix = how_pix(trainData)

#image printing    
    #ax1 : axes object, figsize : image size
fig1, axis = plt.subplots(1,1, figsize=(5,5))
axis.imshow(img[0].reshape(pix))
#plt.xticks([])
#plt.yticks([])


#image printing
fig1, axis = plt.subplots(1,5, figsize=(8,8))
for i in range(10):
    axis[i].imshow(img[i].reshape((pix)))
    axis[i].axis('off')
    axis[i].set_title(label[i])
    
    
    
