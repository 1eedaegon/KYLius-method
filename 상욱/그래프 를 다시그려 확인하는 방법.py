import tensorflow as tf
import numpy as np
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777) 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
%matplotlib inline

#train = pd.read_csv('c:/python/train.csv')
train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])          # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, 1])
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10])
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

# prepare session

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#saver.restore(sess, 'c:/Users/STU/cnn_session')
saver.restore(sess, '/home/itwill03/다운로드/cnn_session')


print('reload has been done')

# multiple images test
result_show = []
fig = plt.figure(figsize=(15,5))
for i in range(0, 9):
    im=Image.open("/home/itwill03/다운로드/numbers_image/number{}.jpeg".format(i+1))
    img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
    data = img.reshape([1, 784])
    data = 1-(data/255)
    ax = fig.add_subplot(1,10,i+1)
    ax.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest', aspect='auto')

    result = sess.run(logits, feed_dict={X:data, training:False})
    result_show.append(sess.run(tf.argmax(result, 1)))
print("MNIST predicted Number")
print(result_show)  

# one image test 

im=Image.open("/home/itwill03/다운로드/numbers_image/number5.jpeg")
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
data = img.reshape([1, 784])
data = 1-(data/255)
plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest')
result = sess.run(logits, feed_dict={X:data, training:False})
print("MNIST predicted Number : ", sess.run(tf.argmax(result, 1)))

sess.close()

