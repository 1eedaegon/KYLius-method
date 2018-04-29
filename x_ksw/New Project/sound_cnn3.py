import numpy as np
np.random.seed(777)

import os
import shutil
import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
train_set, test_set = train_test_split(df, test_size = 0.3)
trainfile = train_set['fname']
testfile = test_set['fname']
trainLabel = train_set.values[:,-1]
testLabel = test_set.values[:,-1]

train.shape[0]
config.dim[0]
config.dim[1]
X = np.empty(shape=(9473, 40, 173, 1)) #  엔트리를 초기화 하지 않고 값을 반환

train.index
train.index[1]

config.audio_length

X_train = prepare_data(train_set, config, '/home/itwill03/sound/audio_train/')
X_test = prepare_data(test_set, config, '/home/itwill03/sound/audio_train/')

#Normalization

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

X_train.shape # (6631, 40, 173, 1)
X_test.shape # (2842, 40, 173, 1)

trainData = X_train.reshape(-1, 40, 173)
testData = X_test.reshape(-1, 40, 173)

trainData.shape # (6631, 40, 173)
testData.shape # (2842, 40, 173)
trainLabel.shape # (6631,)
testLabel.shape # (2842,)

tf.reset_default_graph()     #그래프 초기화

# hyper parameters
learning_rate = 0.0002
training_epochs = 300
batch_size = 100
steps_for_validate = 5

#placeholder
X = tf.placeholder(tf.float32, [None, 40, 173], name="X")
X_sound = tf.reshape(X, [-1, 40, 173, 1])          # 40*173*1 (frequency, time, amplitude)
Y = tf.placeholder(tf.int32, [None, 1], name="Y")
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

# L1 SoundIn shape=(?, 40, 173, 1)
W1 = tf.get_variable("W1", shape=[2, 8, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_sound, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.elu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 8, 1],strides=[1, 2, 8, 1], padding='SAME') 
L1 = tf.nn.dropout(L1, p_keep_conv)

# L2 Input shape=(?,14,58,32)
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME') 
L2 = tf.nn.dropout(L2, p_keep_conv)

# L3 Input shape=(?,5,20,64)
W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME') 
L3 = tf.nn.dropout(L3, p_keep_conv)
L3_flat= tf.reshape(L3, shape=[-1, 3*3*128])

# Final FC 2*3*128 inputs -> 41 outputs
W4 = tf.get_variable("W4", shape=[3*3*128, 512],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.elu(tf.matmul(L3_flat, W4))
L4 = tf.nn.dropout(L4, p_keep_hidden)
W_o = tf.get_variable("W_o", shape=[512,41],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([41]))
logits = tf.matmul(L4, W_o) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1, name="pred")

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver()

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, p_keep_conv: .8, p_keep_hidden: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: testData, Y: testLabel.reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))
        #save_path = saver.save(sess, '/home/paperspace/Downloads/optx/optx')
print('Finished!')


trainData.shape
batch_xs.shape
batch_ys.shape

sess.close()


"""
1) con2d layer * 3 + FC
lr=0.001, epoch = 100    
p_keep_conv, p_keep_hidden = 0.8, 0.7
win : (4, 10, 32), (4, 10, 32), (4, 10, 32)
FC : (?, 64), (64, 41)
max_pool : (2,2), (2,2), (3,3)
accuracy: 51~60%

2) con2d layer * 3 + FC
lr=0.001, epoch = 100    
p_keep_conv, p_keep_hidden = 0.8, 0.7
win : (4, 10, 32), (4, 10, 32), (4, 10, 32)
FC : (2*8*32, 64), (64, 41)
max_pool : (2,2), (2,2), (2,2)
accuracy: 55~60%

3) con2d layer * 3 + FC # 코스트는 계속해서 내려가나 (0.063까지 나옴) 정확도는 그대로임
lr=0.0002, epoch = 300    
p_keep_conv, p_keep_hidden = 0.8, 0.7
win : (4, 10, 32), (4, 10, 32), (4, 10, 32)
FC : (2*8*32, 64), (64, 41)
max_pool : (2,2), (2,2), (2,2)
accuracy: 54~55%

4) con2d layer * 3 + FC # 코스트는 계속해서 내려가나 (0.012까지 나옴) 정확도는 그대로임
lr=0.0002, epoch = 300    
p_keep_conv, p_keep_hidden = 0.8, 0.7
win : (2, 8, 32), (3, 3, 64), (3, 3, 128)
FC : (3*3*128, 512), (512, 41)
max_pool : (2,8), (3,3), (3,3)
accuracy: 61~63%

"""
