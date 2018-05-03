#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:53:10 2018

@author: kimseunghyuck
"""

import librosa
import numpy as np
#import labels
import tensorflow as tf
tf.set_random_seed(777) 
import pandas as pd
train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
#train = pd.read_csv('/home/paperspace/Downloads/audio_train.csv')

#train/test, Data/Label split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.05)
trainfile = train_set.values[:,0]
testfile = test_set.values[:,0]
trainLabel = train_set.values[:,1]
testLabel = test_set.values[:,1]

#data load and extract mfcc (scaling indluded)
path = '/Users/kimseunghyuck/desktop/'
#path = '/home/paperspace/Downloads/'

def see_how_long(file):
    c=[]
    for filename in file:
        y, sr = librosa.core.load(path+filename, mono=True, res_type="kaiser_fast")
        stft=librosa.core.stft(y,1024,512)        
        abs_stft=np.abs(stft)
        #1025 X t 형태
        c.append(abs_stft.shape[1])
    return(c)
 
#n=see_how_long(trainfile)
#print(np.max(n), np.min(n))      #1292 14
#n2=see_how_long(testfile)
#print(np.max(n2), np.min(n2))    #1292 13

def five_sec_extract(file):
    #zero padding to file.shape[0] X 17 X 200    
    n=file.shape[0]
    array = np.repeat(0., n * 17 * 200).reshape(n, 17, 200)
    k=0    
    for filename in file:    
        y, sr = librosa.core.load(path+'audio_train/'+filename, 
                                  mono=True, res_type="kaiser_fast")
        stft=librosa.core.stft(y,32,16)
        mag, pha = librosa.magphase(stft)
        length=stft.shape[1]
        abs_mag=np.abs(mag)
        if length == 200:
            array[k, :, :]=mag
        elif length < 200:
            array[k, :, :length]=mag
        elif length > 200:
            argmax=np.argmax(abs_mag, axis=1)
            sample=[]
            for i in range(np.max(argmax)):
                 sample.append(np.sum((argmax>=i) & (argmax <i+200)))
            start=sample.index(max(sample))
            array[k, :, :]=mag[:, start:start+200]
        k+=1
    return(array)

trainData=five_sec_extract(trainfile)
testData=five_sec_extract(testfile)

print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# (6631, 17, 200) (2842, 17, 200) (6631,) (2842,)

#how many kinds of label?
print(len(np.unique(trainLabel)))   #41
print(len(np.unique(testLabel)))    #41

#label string -> integer(0~40)

def Labeling(label):
    #idx = np.unique(train.values[:,1])     #이건 abc 순
    idx = train.label.unique()
    r=pd.Series(label)
    for i in range(len(idx)):
        r[r.values==idx[i]]=i
    return(r)

trainLabel=Labeling(trainLabel)
testLabel=Labeling(testLabel)
print(min(trainLabel), max(trainLabel), min(testLabel), max(testLabel))
#0 40 0 40

#csv downdload totally about 600MB
trainData2D=trainData.reshape(-1, 17*200)
testData2D=testData.reshape(-1, 17*200)
np.savetxt(path+'trainData7.csv', 
           trainData2D, delimiter=",")
np.savetxt(path+'testData7.csv', 
           testData2D, delimiter=",")
np.savetxt(path+'trainLabel7.csv', 
           trainLabel, delimiter=",")
np.savetxt(path+'testLabel7.csv', 
           testLabel, delimiter=",")
np.savetxt(path+'testfile7.csv', 
           testfile, header = " ", fmt='%s')


