# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 04:28:33 2018

@author: modes
"""

import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt


"""
특성추출
stft(Short-time Fourier transform) :복소수 값을 갖는 행렬을 반환
mfcc(Mel-frequency cepstral coefficients), 
chroma_stft(chromagram from a waveform or power spectrogram), 
melspectrogram(Mel-scaled power spectrogram), 
spectral_contrast(spectral contrast), 
tonnetz(tonal centroid features) 
"""
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name) #부동 소수점 시계열로 로드, sample_rate:자동 리샘플링(default sr=22050) 
    stft = np.abs(librosa.stft(X))  #stft를 절대값으로
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #오디오 신호를 mfcc로 바꿈
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) # stft에서 chromagram 계산
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #melspectrogram
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # spectral_contrast
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0) #tonnetz
    return mfccs,chroma,mel,contrast,tonnetz

#hstack feature matrix
def parse_audio_files(filenames):
    rows = len(filenames)
    features = np.zeros((rows,193))
    i = 0
    for f_names in filenames:
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(f_names)
        features[i] = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        i += 1
    return features    

#file_list 생성
file = "C:/data/sound/audio_train/*.wav"
train_list=glob.glob(file)

file = "c:/data/sound/audio_test/*.wav"
test_list=glob.glob(file)

#extrect audio features
feature_train = parse_audio_files(train_list)
feature_test = parse_audio_files(test_list)

np.savetxt("c:/data/sound/feature_train.csv",feature_train, delimiter=",")


#mel
l = train_list[300]
y,sr = librosa.load(l, duration=2.97)
mel = librosa.feature.melspectrogram(y=y, sr=sr).T #melspectrogram
#mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)
mel.shape #Out[21]: (105, 128)

plt.subplots(1,1, figsize=(5,5))
plt.imshow(mel)

#mfcc
l = train_list[2000]
y,sr = librosa.load(l, duration=2.97)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T #melspectrogram
mfcc.shape #Out[21]: (105, 128)

plt.subplots(1,1, figsize=(5,5))
plt.imshow(mfcc)


#zero padding funtion
def mel2square(mel):    
# 1 zero padded to the top
# 2 zeros padded to the bottom
# 2 zeros padded to the left
# 1 zero padded to the right
    y,x=mel.shape
    return np.pad(mel, ((0,abs(x-y)),(0,0)), 'constant')
mel.shape #Out[96]: (105, 128)
mel2square(mel).shape #Out[95]: (128, 128)

plt.subplots(1,1, figsize=(5,5))
plt.imshow(mel)


plt.subplots(1,1, figsize=(5,5))
plt.imshow(mel2square(mel))


def mfcc2square(mel):    
# 1 zero padded to the top
# 2 zeros padded to the bottom
# 2 zeros padded to the left
# 1 zero padded to the right
    y,x=mel.shape
    return np.pad(mel, ((0,0),(0,abs(x-y))), 'constant')

mfcc.shape #Out[135]: (128, 40)
mfcc2square(mfcc).shape #Out[136]: (128, 128)

plt.subplots(1,1, figsize=(5,5))
plt.imshow(mfcc)
plt.subplots(1,1, figsize=(5,5))
plt.imshow(mfcc2square(mfcc))

#동시 2개 출력??????????????????
fig1, axis =plt.subplots(1,2, figsize=(15,15))
for i in range(2):
    axis[i].imshow(mfcc)
    axis[i].imshow(mfcc2square(mel))




