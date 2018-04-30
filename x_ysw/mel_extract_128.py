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


#zero padding funtion
def mel2square(mel):    
# 1 zero padded to the top
# 2 zeros padded to the bottom
# 2 zeros padded to the left
# 1 zero padded to the right
    y,x=mel.shape
    return np.pad(mel, ((0,0),(0,abs(x-y))), 'constant')
mel.shape #Out[96]: (105, 128)
mel2square(mel).shape #Out[95]: (128, 128)


def Mk_mel(audio_list):
    b= []
    for i in audio_list:
        y,sr = librosa.load(i, duration=2.97)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        res = mel2square(mel)
        b.append(res)
    return np.array(b)
    
a = train_list[0:10]

lists = Mk_mel(a)


