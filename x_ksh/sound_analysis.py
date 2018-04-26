# Load the data and calculate the time of each sample
#samplerate, data = wavfile.read('/Users/kimseunghyuck/desktop/baby.wav')
import librosa
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np

# Get example audio file
# filename = librosa.util.example_audio_file()
filename = '/Users/kimseunghyuck/desktop/baby.wav'

# 1번째 방법
data, samplerate = sf.read(filename, dtype='float32')
data = data.T
data_22k = librosa.resample(data, samplerate, 22050)
plt.figure(figsize=(8, 8))
plt.plot(data_22k[0])
plt.show()

# 2번째 방법
y, sr = librosa.load(filename, duration=10)
p0 = librosa.feature.poly_features(y=y, sr=sr, order=0)
p1 = librosa.feature.poly_features(y=y, sr=sr, order=1)
p2 = librosa.feature.poly_features(y=y, sr=sr, order=2)
plt.figure(figsize=(8, 8))
ax = plt.subplot(4,1,1)
plt.plot(p2[2], label='order=2', alpha=0.8)
plt.plot(p1[1], label='order=1', alpha=0.8)
plt.plot(p0[0], label='order=0', alpha=0.8)
plt.xticks([])
plt.ylabel('Constant')
plt.legend()
plt.subplot(4,1,2, sharex=ax)
plt.plot(p2[1], label='order=2', alpha=0.8)
plt.plot(p1[0], label='order=1', alpha=0.8)
plt.xticks([])
plt.ylabel('Linear')
plt.subplot(4,1,3, sharex=ax)
plt.plot(p2[0], label='order=2', alpha=0.8)
plt.xticks([])
plt.ylabel('Quadratic')
#plt.subplot(4,1,4, sharex=ax)
#librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
#                         y_axis='log')
#plt.tight_layout()

import os
import glob
path='/Users/kimseunghyuck/desktop/audio_train'
files=os.listdir(path)
files[:10]

i=0
y, sr = librosa.load(path+'/'+files[i], duration=10)
p = librosa.feature.poly_features(y=y, sr=sr, order=0)
p=p.reshape(-1,)
plt.figure(figsize=(8, 2))
plt.plot(p)
plt.show()


