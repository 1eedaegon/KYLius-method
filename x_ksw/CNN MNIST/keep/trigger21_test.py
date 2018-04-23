%matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf

tf.reset_default_graph() #그래프 초기화
tf.set_random_seed(777) 

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

LABELS = 10 # 숫자 레이블 (0~9)
WIDTH = 28 # 이미지의 높이와 넓이
CHANNELS = 1 # 이미지 색깔 (white/black)

VALID = 10000 # 유효성 검서 데이터 크기

STEPS = 3500 #20000   # 실행할 단계수
BATCH = 100 # 배치 사이즈
PATCH = 5 # 커널 사이즈
DEPTH = 8 #32 # 컨볼 루션 커널 깊이 크기 == 컨볼 루션 커널 수
HIDDEN = 100 #1024 # 히든 레이어 

LR = 0.001 # Learning rate

data = pd.read_csv('c:/python/train.csv')

# 학습 데이터를 학습데이터와 모의고사 데이터로 나눔

# 1. ver KYLius
labels = np.array(data.pop('label'))
data = StandardScaler().fit_transform(np.float32(data.values))
data = data.reshape(-1, WIDTH, WIDTH, CHANNELS)
train_data, valid_data = train_test_split(data, test_size = 0.3)

#one hot function(n=10)
def one_hot(x):
    lst=[]
    for i in x:
        lst.append([0,0,0,0,0,0,0,0,0,0])
        lst[-1][i]+=1
    return(lst)

train_labels, valid_labels = train_test_split(labels, test_size = 0.3)
train_labels=one_hot(train_labels)
valid_labels=one_hot(valid_labels)

print('train data shape = ' + str(train_data.shape) + ' = (TRAIN, WIDTH, WIDTH, CHANNELS)')
print('labels shape = ' + str(labels.shape) + ' = (TRAIN, LABELS)')


               # or

# 2. ver Trigger
labels = np.array(data.pop('label')) # 레이블 분리
labels = LabelEncoder().fit_transform(labels)[:, None]
labels = OneHotEncoder().fit_transform(labels).todense()
data = StandardScaler().fit_transform(np.float32(data.values)) # 데이터 프레임 배열로 변환
data = data.reshape(-1, WIDTH, WIDTH, CHANNELS) # 데이터를 42000 2차원 이미지로 변환

train_data, valid_data = data[:-VALID], data[-VALID:]
train_labels, valid_labels = labels[:-VALID], labels[-VALID:]

print('train data shape = ' + str(train_data.shape) + ' = (TRAIN, WIDTH, WIDTH, CHANNELS)')
print('labels shape = ' + str(labels.shape) + ' = (TRAIN, LABELS)')


# 플레이스 홀더로 빈공간의 데이터 생성
tf_data = tf.placeholder(tf.float32, shape=(None, WIDTH, WIDTH, CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABELS))


# HIDDEN 숨겨진 뉴런과 출력 레이어 (w4, b3)가 완전히 연결된 숨겨진 레이어 (w3, b3)가 뒤 따르는 가중치 및 바이어스 (w1, b1) 및 (w2, b2)가있는 2 개의 길쌈 레이어 , b4) 10 출력 노드 (one-hot 인코딩).

# 커널이 패치 크기가 PATCH이고 두 번째 컨 벌루 셔널 레이어의 깊이가 첫 번째 컨볼 루션 레이어 (DEPTH)의 두 배가되도록 가중치와 바이어스를 초기화합니다. 나머지의 경우, 완전히 연결된 숨겨진 레이어에는 HIDDEN 숨겨진 뉴런이 있습니다.

w1 = tf.Variable(tf.truncated_normal([PATCH, PATCH, CHANNELS, DEPTH], stddev=0.1))
b1 = tf.Variable(tf.zeros([DEPTH]))
w2 = tf.Variable(tf.truncated_normal([PATCH, PATCH, DEPTH, 2*DEPTH], stddev=0.1))
b2 = tf.Variable(tf.constant(1.0, shape=[2*DEPTH]))
w3 = tf.Variable(tf.truncated_normal([WIDTH // 4 * WIDTH // 4 * 2*DEPTH, HIDDEN], stddev=0.1))
b3 = tf.Variable(tf.constant(1.0, shape=[HIDDEN]))
w4 = tf.Variable(tf.truncated_normal([HIDDEN, LABELS], stddev=0.1))
b4 = tf.Variable(tf.constant(1.0, shape=[LABELS]))

def logits(data):
    # Convolutional layer 1
    x = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b1)
    # Convolutional layer 2
    x = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b2)
    # Fully connected layer
    x = tf.reshape(x, (-1, WIDTH // 4 * WIDTH // 4 * 2*DEPTH))
    x = tf.nn.relu(tf.matmul(x, w3) + b3)
    return tf.matmul(x, w4) + b4

# Prediction:
tf_pred = tf.nn.softmax(logits(tf_data))

# 로스값은 에폭과 배치의 값에 따라
# 옵티마이저는 그라디언트 하강 옵티마이저 (학습 속도 감소 또는 감소없이) 또는 Adam 또는 RMSProp과 같은보다 정교하고 최적화하기 쉬운 옵티 마이저 중 하나를 사용할 수 있습니다

tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits(tf_data),labels=tf_labels))
tf_acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_labels, 1))))

tf_opt = tf.train.GradientDescentOptimizer(LR) # 그라디언트 옵티
tf_opt = tf.train.AdamOptimizer(LR) # 아담 옵티
#tf_opt = tf.train.RMSPropOptimizer(LR) # RMS 옵티
tf_step = tf_opt.minimize(tf_loss)

# 세션 수행

init = tf.global_variables_initializer() # 글로벌 변수 초기화
session = tf.Session()
session.run(init)

ss = ShuffleSplit(n_splits=STEPS, train_size=BATCH)
ss.get_n_splits(train_data, train_labels)
history = [(0, np.nan, 10)] # 오류 난거 수정

for step, (idx, _) in enumerate(ss.split(train_data,train_labels), start=1):
    fd = {tf_data:train_data[idx], tf_labels:train_labels[idx]}
    session.run(tf_step, feed_dict=fd)
    if step%500 == 0:
        fd = {tf_data:valid_data, tf_labels:valid_labels}
        valid_loss, valid_accuracy = session.run([tf_loss, tf_acc], feed_dict=fd)
        history.append((step, valid_loss, valid_accuracy))
        print('Step %i \t Valid. Acc. = %f'%(step, valid_accuracy), end='\n')
        
steps, loss, acc = zip(*history)

fig = plt.figure()
plt.title('Validation Loss / Accuracy')
ax_loss = fig.add_subplot(111)
ax_acc = ax_loss.twinx()
plt.xlabel('Training Steps')
plt.xlim(0, max(steps))

ax_loss.plot(steps, loss, '-o', color='C0')
ax_loss.set_ylabel('Log Loss', color='C0');
ax_loss.tick_params('y', colors='C0')
ax_loss.set_ylim(0.01, 0.5)

ax_acc.plot(steps, acc, '-o', color='C1')
ax_acc.set_ylabel('Accuracy [%]', color='C1');
ax_acc.tick_params('y', colors='C1')
ax_acc.set_ylim(1,100)

plt.show()

# test

test = pd.read_csv('c:/python/test.csv')
test_data = StandardScaler().fit_transform(np.float32(test.values)) # 배열로 변환
test_data = test_data.reshape(-1, WIDTH, WIDTH, CHANNELS) # 트레인 셋과 마찬가지로 42000 2차원 이미지로 변환

test_pred = session.run(tf_pred, feed_dict={tf_data:test_data})
test_labels = np.argmax(test_pred, axis=1)

k = 0 # 다른이미지 색인 시도
print("Label Prediction: %i"%test_labels[k])
fig = plt.figure(figsize=(2,2)); plt.axis('off')
plt.imshow(test_data[k,:,:,0]); plt.show()

session.close()
