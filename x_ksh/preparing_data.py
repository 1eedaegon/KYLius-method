#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:53:10 2018

@author: kimseunghyuck
"""

import librosa
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
#import labels
import tensorflow as tf
tf.set_random_seed(777) 
import pandas as pd
train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
#train = pd.read_csv('/home/paperspace/Downloads/audio_train.csv')

#train/test, Data/Label split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.3)
trainfile = train_set.values[:,0]
testfile = test_set.values[:,0]
trainLabel = train_set.values[:,1]
testLabel = test_set.values[:,1]

#data load and extract mfcc (scaling indluded)
path = '/Users/kimseunghyuck/desktop/audio_train/'
#path = '/home/paperspace/Downloads/audio_train/'

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

#show me approximate wave shape
filename= trainfile[11]
y, sr = sf.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
length=mfcc.shape[1]
plt.plot(mfcc[3,])
plt.plot(np.abs(mfcc[3,]))
plt.plot(mfcc[2,])
plt.plot(np.abs(mfcc[3,]))

#5 seconds(430 segments) extract
def five_sec_extract(file):
    #zero padding to file.shape[0] X 40 X 3000    
    n=file.shape[0]
    array = np.repeat(0, n * 20 * 430).reshape(n, 20, 430)
    k=0    
    for filename in file:    
        y, sr = sf.read(path+filename, dtype='float32')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        length=mfcc.shape[1]
        abs_mfcc=np.abs(mfcc)
        if length == 430:
            array[k, :, :]=mfcc
        elif length < 430:
            array[k, :, :length]=mfcc
        elif length > 430:
            argmax=np.argmax(abs_mfcc, axis=1)
            sample=[]
            for i in range(np.max(argmax)):
                sample.append(np.sum((argmax>=i) & (argmax <i+430)))
            start=sample.index(max(sample))
            array[k, :, :]=mfcc[:, start:start+430]
        k+=1
    array=array.reshape(-1,20*430)
    array=((array-np.mean(array))/np.std(array)).reshape(-1,20,430)
    return(array)

trainData=five_sec_extract(trainfile)
testData=five_sec_extract(testfile)


print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 20, 430) (2842, 20, 430) (6631,) (2842,)

#how many kinds of label?
print(len(np.unique(trainLabel)))   #41
print(len(np.unique(testLabel)))    #41

#label string -> integer(0~40)

def Labeling(label):
    idx = np.unique(train.values[:,1])
    r=pd.Series(label)
    for i in range(len(idx)):
        r[r.values==idx[i]]=i
    return(r)

trainLabel=Labeling(trainLabel)
testLabel=Labeling(testLabel)
print(min(trainLabel), max(trainLabel), min(testLabel), max(testLabel))
#0 40 0 40


#csv downdload totally about 600MB
trainData2D=trainData.reshape(-1, 20*430)
testData2D=testData.reshape(-1, 20*430)
np.savetxt('/Users/kimseunghyuck/desktop/trainData4.csv', 
           trainData2D, delimiter=",")
np.savetxt('/Users/kimseunghyuck/desktop/testData4.csv', 
           testData2D, delimiter=",")
np.savetxt('/Users/kimseunghyuck/desktop/trainLabel4.csv', 
           trainLabel, delimiter=",")
np.savetxt('/Users/kimseunghyuck/desktop/testLabel4.csv', 
           testLabel, delimiter=",")

