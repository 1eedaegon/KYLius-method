# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:48:54 2018

@author: stu
"""

import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

#file_list 생성
file = "C:/data/sound/audio_train/*.wav"
train_list=glob.glob(file)

file = "c:/data/sound/audio_test/*.wav"
test_list=glob.glob(file)


#zero padding funtion : 128보다 작은경우 128*128로 zero padding 
def mel2square(mel):    
# 1 zero padded to the top
# 2 zeros padded to the bottom
# 2 zeros padded to the left
# 1 zero padded to the right
    y,x=mel.shape
    return np.pad(mel, ((0,0),(0,abs(x-y))), 'constant')
mel.shape #Out[96]: (105, 128)
mel2square(mel).shape #Out[95]: (128, 128)

# extract melspectrogram with padding : 'duration=2.97'로 설정해서 y축길이의 max값을 128로 고정
def Mk_mel(audio_list):
    b= []
    for i in audio_list:
        y,sr = librosa.load(i, duration=2.97)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        res = mel2square(mel)
        b.append(res)
        print(i)
    return np.array(b)
    

lists = Mk_mel(train_list)

#uploading csv file : shape = [?,128,128]
lists2D=np.reshape(lists,[-1,16384])
np.savetxt('C:/data/sound/mel_train4.csv',lists2D, delimiter=',')

mel_train=np.genfromtxt('C:/data/sound/mel_train5.csv', delimiter=',')








