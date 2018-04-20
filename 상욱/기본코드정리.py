import tensorflow as tf
import numpy as np
tf.reset_default_graph()     # 그래프 초기화
tf.set_random_seed(777)      # 값이 랜덤하게 들어가더라도 일정하게 전달되게 적용
import pandas as pd

# 학습데이터 불러 오기
#train = pd.read_csv('c:/python/train.csv')
train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3) #학습데이터, 테스트 데이터 7:3 비율로 나눔
trainData = train_set.values[:,1:] # 학습용 라벨을 제외한 행 입력
validateData = validate_set.values[:,1:] # 테스트 라벨을 제외한 행 입력
trainLabel=train_set.values[:,0] # 트레인 라벨 입력
validateLabel=validate_set.values[:,0] # 테스트 라벨 입력

X = tf.placeholder(tf.float32, [None, 784]) # 라벨행을 제외한 784개 행의 공간
X_img = tf.reshape(X, [-1, 28, 28, 1])          # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, 1]) # 라벨값의 공간
Y_onehot=tf.reshape(tf.one_hot(Y, 10), [-1, 10]) # 원핫 인코딩을 통해 값을 바꿈 대신 원핫 인코딩으로 인한 3차원 배열로 바뀐 값을 2차원으로 다시 바꿔줌 

# hyper parameters
learning_rate = 0.001 # 러닝레이트 값
training_epochs = 20 # 트레이닝 에폭값
batch_size = 100 # 베치사이즈
steps_for_validate = 5 # 스텝 입력값


# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l1 shape=(?, 14, 14, 32)


# L2 ImgIn shape=(?, 14, 14, 10)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)


# L3 ImgIn shape=(?, 7, 7, 128)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l3 shape=(?, 4, 4, 128)
L3_flat = tf.reshape(L3, shape=[-1, 128 * 4 * 4])    # reshape to (?, 2048)


# Final FC 4x4x128 inputs -> 10 outputs
W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
W_o = tf.get_variable("W_o", shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W_o) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
saver = tf.train.Saver() # 옵티마이져 저장을 위한 코드

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size)
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1)}))       
print('Finished!')

saver.save(sess, '/home/itwill03/다운로드/cnn_session')
save_path = saver.save(sess, "/home/itwill03/다운로드//opt2")

sess.close()