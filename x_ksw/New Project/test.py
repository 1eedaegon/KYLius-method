# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 23:59:42 2018

@author: modes
"""
import os
import glob
import librosa
import numpy as np
import pandas as pd
28*28

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
    #stft = np.abs(librosa.stft(X))  #stft를 절대값으로
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=784).T,axis=0) #오디오 신호를 mfcc로 바꿈
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) # stft에서 chromagram 계산
    #mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #melspectrogram
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # spectral_contrast
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0) #tonnetz
    #return mfccs,chroma,mel,contrast,tonnetz
    return mfccs
#hstack feature matrix
def parse_audio_files(filenames):
    rows = len(filenames)
    features = np.zeros((rows,784))
    i = 0
    for f_names in filenames:
        mfccs = extract_feature(f_names)
        features[i] = np.hstack(mfccs)
        i += 1
    return features    

#file_list 생성
file = "/home/itwill03/sound/audio_train/*.wav"
train_list=glob.glob(file)

file = "/home/itwill03/sound/audio_test/*.wav"
test_list=glob.glob(file)
test_list[1414]
len(train_list)

#extrect audio features
feature_train = parse_audio_files(train_list)
feature_test = parse_audio_files(test_list)

sf = "/home/itwill03/sound/audio_train/0a0a8d4c.wav"
sf = train_list[4100]
X, sample_rate = librosa.load(sf,sr=22050)
mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T
plt.subplots(1,1, figsize=(5,5))
plt.imshow(mfccs)

feature_train = extract_feature(train_list)
feature_test = extract_feature(test_list)

np.savetxt("/home/itwill03/sound/feature_train.csv",feature_train, delimiter=",")


len(feature_train)
len(train_list)

feature_train.shape
feature_test.shape

train = pd.read_csv("/home/itwill03/sound/train.csv")
ft = pd.read_csv("/home/itwill03/sound/feature_train.csv", header = None)
ft2 = pd.DataFrame(feature_train)
ft.loc[0]

df = pd.concat([ft, train['label']], axis=1)

train.head()
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))
print(train.label.unique())

for i in range(len(feature_train)+1):
    train['mfcc'][i] = feature_train[i]
    
feature_train[0]

train['fname'][0]
train['label']
a = extract_feature('/home/itwill03/sound/audio_test/87f52da2.wav')
b = extract_feature('/home/itwill03/sound/audio_test/1819a7b7.wav') 


len(a[0])
len(b[1])

feature_train = np.zeros((len(train_list),40))
i = 0
for f_name in train_list:    
    feature_train[i] = extract_feature(f_name)
    i += 1
feature_train


40,


df
df['label'][0]
print(train.label.unique())

if df['label'][0] == 'Hi-hat':
    df['lb'] = 0

for i in range(len(df)):
    if df['label'][i] == 'Hi-hat':
        df['lb'][i] = 0
    elif df['label'][i] == 'Saxophone':
        df['lb'][i] = 1
    elif df['label'][i] == 'Trumpet':
        df['lb'][i] = 2
    elif df['label'][i] == 'Glockenspiel':
        df['lb'][i] = 3
    elif df['label'][i] == 'Cello':
        df['lb'][i] = 4
    elif df['label'][i] == 'Knock':
        df['lb'][i] = 5
    elif df['label'][i] == 'Gunshot_or_gunfire':
        df['lb'][i] = 6
    elif df['label'][i] == 'Clarinet':
        df['lb'][i] = 7
    elif df['label'][i] == 'Computer_keyboard':
        df['lb'][i] = 8
    elif df['label'][i] == 'Keys_jangling':
        df['lb'][i] = 9
    elif df['label'][i] == 'Snare_drum':
        df['lb'][i] = 10
    elif df['label'][i] == 'Writing':
        df['lb'][i] = 11
    elif df['label'][i] == 'Laughter':
        df['lb'][i] = 12
    elif df['label'][i] == 'Tearing':
        df['lb'][i] = 13
    elif df['label'][i] == 'Fart':
        df['lb'][i] = 14
    elif df['label'][i] == 'Oboe':
        df['lb'][i] = 15
    elif df['label'][i] == 'Flute':
        df['lb'][i] = 16
    elif df['label'][i] == 'Cough':
        df['lb'][i] = 17
    elif df['label'][i] == 'Telephone':
        df['lb'][i] = 18
    elif df['label'][i] == 'Bark':
        df['lb'][i] = 19
    elif df['label'][i] == 'Chime':
        df['lb'][i] = 20
    elif df['label'][i] == 'Bass_drum':
        df['lb'][i] = 21
    elif df['label'][i] == 'Bus':
        df['lb'][i] = 22
    elif df['label'][i] == 'Squeak':
        df['lb'][i] = 23
    elif df['label'][i] == 'Scissors':
        df['lb'][i] = 24
    elif df['label'][i] == 'Harmonica':
        df['lb'][i] = 25
    elif df['label'][i] == 'Gong':
        df['lb'][i] = 26
    elif df['label'][i] == 'Microwave_oven':
        df['lb'][i] = 27
    elif df['label'][i] == 'Burping_or_eructation':
        df['lb'][i] = 28
    elif df['label'][i] == 'Double_bass':
        df['lb'][i] = 29
    elif df['label'][i] == 'Shatter':
        df['lb'][i] = 30
    elif df['label'][i] == 'Fireworks':
        df['lb'][i] = 31
    elif df['label'][i] == 'Tambourine':
        df['lb'][i] = 32
    elif df['label'][i] == 'Cowbell':
        df['lb'][i] = 33
    elif df['label'][i] == 'Electric_piano':
        df['lb'][i] = 34
    elif df['label'][i] == 'Meow':
        df['lb'][i] = 35
    elif df['label'][i] == 'Drawer_open_or_close':
        df['lb'][i] = 36
    elif df['label'][i] == 'Applause':
        df['lb'][i] = 37
    elif df['label'][i] == 'Acoustic_guitar':
        df['lb'][i] = 38
    elif df['label'][i] == 'Violin_or_fiddle':
        df['lb'][i] = 39
    elif df['label'][i] == 'Finger_snapping':
        df['lb'][i] = 40
    
np.savetxt("/home/itwill03/sound/df_train.csv",df, delimiter=",")


import tensorflow as tf
import numpy as np
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777) 
import pandas as pd

#train = pd.read_csv('c:/python/train.csv')
train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(df, test_size = 0.3)
trainData = train_set.values[:,:-1]
validateData = validate_set.values[:,:-1]
trainLabel=train_set.values[:,-1]
validateLabel=validate_set.values[:,-1]

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])          # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

# hyper parameters
learning_rate = 0.00008
training_epochs = 300
batch_size = 100
steps_for_validate = 5
keep_prob = tf.placeholder(tf.float32)

# L1 ImgIn shape=(?, 28, 28, 1)
#W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#L1 = tf.nn.relu(L1)
#L1 = tf.nn.elu(L1)
L1 = tf.nn.leaky_relu(L1,0.1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l1 shape=(?, 14, 14, 32)
L1 = tf.nn.dropout(L1, p_keep_conv)

# L2 ImgIn shape=(?, 14, 14, 10)
#W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#L2 = tf.nn.relu(L2)
#L2 = tf.nn.elu(L2)
L2 = tf.nn.leaky_relu(L2,0.1)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)
L2 = tf.nn.dropout(L2, p_keep_conv)

# L3 ImgIn shape=(?, 7, 7, 128)
#W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#L3 = tf.nn.relu(L3)
#L3 = tf.nn.elu(L3)
L3 = tf.nn.leaky_relu(L3,0.1)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l3 shape=(?, 4, 4, 128)
L3 = tf.nn.dropout(L3, p_keep_conv)
L3_flat = tf.reshape(L3, shape=[-1, 128 * 4 * 4])    # reshape to (?, 2048)


# Final FC 4x4x128 inputs -> 10 outputs
#W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],initializer=tf.contrib.layers.xavier_initializer())
#L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
#L4 = tf.nn.elu(tf.matmul(L3_flat, W4))
L4 = tf.nn.leaky_relu(tf.matmul(L3_flat, W4),0.1)
L4 = tf.nn.dropout(L4, p_keep_hidden)
W_o = tf.get_variable("W_o", shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W_o) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
#optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, p_keep_conv: .7, p_keep_hidden: .5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))
        saver.save(sess, '/home/itwill03/다운로드/cnn_session/cnn_session')
        save_path = saver.save(sess, "/home/itwill03/다운로드/opt2/opt2")
print('Finished!')

#saver.save(sess, 'cnn_session')
#save_path = saver.save(sess, "c:/Users/STU/opt2")


sess.close()
