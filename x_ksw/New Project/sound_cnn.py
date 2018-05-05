import numpy as np
np.random.seed(1001)

import os
import shutil
tf.set_random_seed(777) 
import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.cross_validation import StratifiedKFold

%matplotlib inline
matplotlib.style.use('ggplot')

# Preparing data

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

config = Config(sampling_rate=44100, audio_duration=2, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)

def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df['fname']):
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X

file = "/home/itwill03/sound/audio_train/*.wav"
train_list=glob.glob(file)

file = "/home/itwill03/sound/audio_test/*.wav"
test_list=glob.glob(file)

train = pd.read_csv("/home/itwill03/sound/train.csv")

labels = train['label']
l = train['label'].unique()

df_label = pd.DataFrame(labels)

for i in range(len(l)):
    df_label[df_label==l[i]] = int(i)
df_label.values

df = pd.concat([train, df_label], axis=1)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.3)
trainfile = train_set['fname']
testfile = test_set['fname']
trainLabel = train_set.values[:,-1]
testLabel = test_set.values[:,-1]



train.shape[0]
config.dim[0]
config.dim[1]
X = np.empty(shape=(9473, 40, 173, 1)) #  엔트리를 초기화 하지 않고 값을 반환

train.index

import IPython.display as ipd  # To play sound in the notebook
fname = '/home/itwill03/sound/audio_train/' + '00044347.wav'   # Hi-hat
ipd.Audio(fname)

for i, fname in enumerate(train['fname']):
    print(i)
    file_path = fname
    data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

train.index[1]

config.audio_length

X_train = prepare_data(train_set, config, '/home/itwill03/sound/audio_train/')
X_test = prepare_data(test_set, config, '/home/itwill03/sound/audio_train/')
y_train = to_categorical(train.label_idx, num_classes=config.n_classes)


#Normalization

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

X_train.shape
X_test.shape
