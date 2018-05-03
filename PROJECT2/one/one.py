import numpy as np
import tensorflow as tf
np.random.seed(1001)
tf.set_random_seed(1001)
import os
import pandas as pd
import seaborn as sns
import librosa


# Preparing data

#  *2.03 : 350

class Config(object): # mfcc에 대한 데이터 길이 정제 작업
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

        self.audio_length = self.sampling_rate * self.audio_duration*2.03
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

config = Config(sampling_rate=44100, audio_duration=2, n_folds=10, # 주파수와 채널 , mfcc 값 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)

# 데이터 길이에 대한 패딩
def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = int(config.audio_length)
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


train = pd.read_csv("c:/sound/train.csv")
#train = pd.read_csv("/home/itwill03/sound/train.csv")

# 라벨 숫자화 작업
labels = train['label']
l = train['label'].unique()

df_label = pd.DataFrame(labels)

for i in range(len(l)):
    df_label[df_label==l[i]] = int(i)
df_label.values

df = pd.concat([train, df_label], axis=1)

df1 = df[df['manually_verified']==1]

# 데이터를 7 : 3 비율로 나눔
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df1, test_size = 0.3)
trainfile = train_set['fname']
testfile = test_set['fname']
trainLabel = train_set.values[:,-1]
testLabel = test_set.values[:,-1]

# 파일 nfcc 화
X_train = prepare_data(train_set, config, 'c:/sound/audio_train/')
X_test = prepare_data(test_set, config, 'c:/sound/audio_train/')

#X_train = prepare_data(train_set, config, '/home/itwill03/sound/audio_train/')
#X_test = prepare_data(test_set, config, '/home/itwill03/sound/audio_train/')

#Normalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

X_train.shape # (6631, 40, 350, 1)
X_test.shape # (2842, 40, 350, 1)
trainLabel.shape # (6631,)
testLabel.shape # (2842,)

# start
tf.reset_default_graph()     #그래프 초기화

# hyper parameters
learning_rate = 0.00008
training_epochs = 700
batch_size = 100
steps_for_validate = 5

#placeholder
X = tf.placeholder(tf.float32, [None, 40, 350,1], name="X") # 40*350*1 (frequency, time, amplitude)
Y = tf.placeholder(tf.int32, [None, 1], name="Y")
Y_onehot=tf.reshape(tf.one_hot(Y, 41), [-1, 41])
p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

# L1 SoundIn shape=(?, 40, 350, 1)
W1 = tf.get_variable("W1", shape=[4, 35, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.layers.batch_normalization(L1)
#L1 = tf.nn.relu(L1)
#L1 = tf.nn.elu(L1)
L1 = tf.nn.leaky_relu(L1,0.1)
L1 = tf.nn.max_pool(L1, ksize=[1, 4, 35, 1],strides=[1, 4, 35, 1], padding='SAME') 
L1 = tf.nn.dropout(L1, p_keep_conv)

# L2 Input shape=(?,10,10,32)
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.layers.batch_normalization(L2)
#L2 = tf.nn.relu(L2)
#L2 = tf.nn.elu(L2)
L2 = tf.nn.leaky_relu(L2,0.1)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
#L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
L2 = tf.nn.dropout(L2, p_keep_conv)

# L3 Input shape=(?,5,5,64)
W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.layers.batch_normalization(L3)
#L3 = tf.nn.relu(L3)
#L3 = tf.nn.elu(L3)
L3 = tf.nn.leaky_relu(L3,0.1)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID') 
L3 = tf.nn.dropout(L3, p_keep_conv)
L3_flat= tf.reshape(L3, shape=[-1, 3*3*128])

# Final FC 2*2*128 inputs -> 41 outputs
W4 = tf.get_variable("W4", shape=[3*3*128, 625],initializer=tf.contrib.layers.xavier_initializer())
L3_flat = tf.layers.batch_normalization(L3_flat)
#L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
#L4 = tf.nn.elu(tf.matmul(L3_flat, W4))
L4 = tf.nn.leaky_relu(tf.matmul(L3_flat, W4),0.1)
L4 = tf.nn.dropout(L4, p_keep_hidden)
W_o = tf.get_variable("W_o", shape=[625,41],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([41]))
logits = tf.matmul(L4, W_o) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1, name="pred")

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(X_train) / batch_size)
    for i in range(total_batch):
        batch_xs = X_train[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, p_keep_conv: 0.7, p_keep_hidden: 0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        x=np.random.choice(testLabel.shape[0], 500, replace=False)
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: X_test[x], Y: testLabel[x].reshape(-1, 1), p_keep_conv: 1, p_keep_hidden: 1}))
    save_path = saver.save(sess, 'c:/sound/optx/optx')
print('Finished!')


sess.close()

"""
1) con2d layer * 3 + FC
lr=0.00008, epoch = 700    
p_keep_conv, p_keep_hidden = 0.7, 0.8
win : (4, 35, 32), (3, 3, 64), (3, 3, 128)
FC : (3*3*128, 625), (625, 41)
max_pool : (4,35), (2,2), (2,2)
accuracy: 69~73%

"""