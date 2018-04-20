import tensorflow as tf
import numpy as np
import pandas as pd
tf.reset_default_graph()     # 그래프 초기화
tf.set_random_seed(777)      # 값이 랜덤하게 들어가더라도 일정하게 전달되게 적용


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
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # 3*3 크기의 1(흑백)의 필터를 32개 만듬
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') # 필터를 1,1 크기로 움직임 이미지 크기를 유지하기 위해서 padding='SAME'
L1 = tf.nn.relu(L1) # 활성화 함수로 값 출력
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # maxpool 을 이용한 샘플링, padding='SAME'을 유지하지만 strides가 2,2 때문에 l1 shape=(?, 14, 14, 32)


# L2 ImgIn shape=(?, 14, 14, 10)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 3*3 크기의 32개의 필터를 받아 64개로 만듬
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # 필터를 1,1 크기로 움직임 이미지 크기를 유지하기 위해서 padding='SAME'
L2 = tf.nn.relu(L2) # 활성화 함수로 값 출력
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # maxpool 을 이용한 샘플링, padding='SAME'을 유지하지만 strides가 2,2 때문에 l2 shape=(?, 7, 7, 64)


# L3 ImgIn shape=(?, 7, 7, 128)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)) # 3*3 크기의 64개의 필터를 받아 128개로 만듬
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') # 필터를 1,1 크기로 움직임 이미지 크기를 유지하기 위해서 padding='SAME'
L3 = tf.nn.relu(L3) # 활성화 함수로 값 출력
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # maxpool 을 이용한 샘플링, padding='SAME'을 유지하지만 strides가 2,2 때문에 l3 shape=(?, 4, 4, 128)
L3_flat = tf.reshape(L3, shape=[-1, 128 * 4 * 4])  # FC로 가기위해 값을 펴줌 reshape to (?, 2048)


# Final FC(Fully connected) 4x4x128 inputs -> 10 outputs
W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01)) # 2048개의 값을 625개로 만듬
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)) 
W_o = tf.get_variable("W_o", shape=[625,10],initializer=tf.contrib.layers.xavier_initializer()) # 입력층을 기준으로 루트를 씌운것 분에 1을 가중치로 두는 인티져
b = tf.Variable(tf.random_normal([10])) # 바이오스값
logits = tf.matmul(L4, W_o) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= Y_onehot)) # 소프트맥스를 이용한 오차값
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담버젼
predict_op = tf.argmax(logits, 1) # 가장 정확도가 높은값

# initialize
sess = tf.Session() # 세션 오픈
sess.run(tf.global_variables_initializer()) # 세션 시작시 텐서 변수에 있는값 초기화
saver = tf.train.Saver() # 옵티마이져 저장을 위한 코드

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(trainData) / batch_size) # 트레인 데이터를 베치 사이즈로 나누어 1에폭까지의 값을 구함
    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size] # 배치사이즈 대로 값이 들어가게 정의
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1) # 펼쳐져 있는 값을 열로 바꾸어줌
        feed_dict = {X: batch_xs, Y: batch_ys} # feed_dict 값을 미리 정의
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict) # 학습을 하며 코스트를 같이 실행 값을 불러옴
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost)) # 에폭당 코스트 출력
    if epoch % steps_for_validate == steps_for_validate-1:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1)) # 가장 정확도가 높은 값과 라벨을 비교
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도를 구함
        print('Accuracy:', sess.run(accuracy, feed_dict={
                X: validateData, Y: validateLabel.reshape(-1, 1)}))   # 테스트 데이터를 비교해 정확도를 측정
print('Finished!')

#saver.save(sess, '/home/itwill03/다운로드/cnn_session')
save_path = saver.save(sess, "/home/itwill03/다운로드//opt2") # 학습된 세션을 저장

sess.close()