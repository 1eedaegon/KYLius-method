# Change this to True to replicate the result
COMPLETE_RUN = False

# Loading data
import numpy as np
np.random.seed(1001)

import os
import shutil

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.cross_validation import StratifiedKFold

%matplotlib inline
matplotlib.style.use('ggplot')

train = pd.read_csv("../input/freesound-audio-tagging/train.csv")
test = pd.read_csv("../input/freesound-audio-tagging/sample_submission.csv")

train.head()

print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))
print(train.label.unique())

# Distribution of Categories
category_group = train.groupby(['label', 'manually_verified']).count()
plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index)\
          .plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16,10))
plot.set_xlabel("Category")
plot.set_ylabel("Number of Samples");

print('Minimum samples per category = ', min(train.label.value_counts()))
print('Maximum samples per category = ', max(train.label.value_counts()))

# Let's listen to an audio file in our dataset and load it to a numpy array
import IPython.display as ipd  # To play sound in the notebook
fname = '../input/freesound-audio-tagging/audio_train/' + '00044347.wav'   # Hi-hat
ipd.Audio(fname)

# Using wave library
import wave
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())

# Using scipy
from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)

# Let's plot the audio frames
plt.plot(data, '-', );

# Let's zoom in on first 1000 frames
plt.figure(figsize=(16, 4))
plt.plot(data[:500], '.'); plt.plot(data[:500], '-');

# Audio Length
# We shall now analyze the lengths of the audio files in our dataset
train['nframes'] = train['fname'].apply(lambda f: wave.open('../input/freesound-audio-tagging/audio_train/' + f).getnframes())
test['nframes'] = test['fname'].apply(lambda f: wave.open('../input/freesound-audio-tagging/audio_test/' + f).getnframes())

_, ax = plt.subplots(figsize=(16, 4))
sns.violinplot(ax=ax, x="label", y="nframes", data=train)
plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.show()

# Let's now analyze the frame length distribution in Train and Test.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
train.nframes.hist(bins=100, ax=axes[0])
test.nframes.hist(bins=100, ax=axes[1])
plt.suptitle('Frame Length Distribution in Train and Test', ha='center', fontsize='large');

"""
We observe:

Majority of the audio files are short.
There are four abnormal length in the test histogram. Let's analyze them.
"""

abnormal_length = [707364, 353682, 138474, 184338]

for length in abnormal_length:
    abnormal_fnames = test.loc[test.nframes == length, 'fname'].values
    print("Frame length = ", length, " Number of files = ", abnormal_fnames.shape[0], end="   ")
    fname = np.random.choice(abnormal_fnames)
    print("Playing ", fname)
    IPython.display.display(ipd.Audio( '../input/freesound-audio-tagging/audio_test/' + fname))
    
