import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
tf.reset_default_graph()     #그래프 초기화
tf.set_random_seed(777) 

train = pd.read_csv('/home/itwill03/다운로드/train.csv')

from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
trainData = train_set.values[:,1:]
validateData = validate_set.values[:,1:]
trainLabel=train_set.values[:,0]
validateLabel=validate_set.values[:,0]

learning_rate = 0.00007
training_epochs = 500
batch_size = 100
steps_for_validate = 10



class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        # 입력 받은 이름으로 변수 명을 설정한다.
        with tf.variable_scope(self.name):

            # Boolean Tensor 생성 for dropout
            # tf.layers.dropout( training= True/Fals) True/False에 따라서 학습인지 / 예측인지 선택하게 됨
            # default = False
            self.p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
            self.p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

            # 입력 그래프 생성
            self.X = tf.placeholder(tf.float32, [None, 784], name="X")
            # 28x28x1로 사이즈 변환
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.int32, [None, 1], name="Y")
            Y_onehot=tf.reshape(tf.one_hot(self.Y, 10), [-1, 10])

            # Convolutional Layer1
            W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            #L1 = tf.nn.elu(L1)
            L1 = tf.nn.leaky_relu(L1,0.1)
            # Pooling Layer1
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
            # Dropout Layer1
            L1 = tf.nn.dropout(L1, self.p_keep_conv)


            # Convolutional Layer2
            W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            #L2 = tf.nn.relu(L2)
            #L2 = tf.nn.elu(L2)
            L2 = tf.nn.leaky_relu(L2,0.1)
            # Pooling Layer2
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)
            # Dropout Layer2
            L2 = tf.nn.dropout(L2, self.p_keep_conv)


            # Convolutional Layer3
            W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            #L3 = tf.nn.relu(L3)
            #L3 = tf.nn.elu(L3)
            L3 = tf.nn.leaky_relu(L3,0.1)
            # Pooling Layer3
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l3 shape=(?, 4, 4, 128)
            # Dropout Layer3
            L3 = tf.nn.dropout(L3, self.p_keep_conv)
            L3_flat = tf.reshape(L3, shape=[-1, 128 * 4 * 4])  

            # Dense Layer4 
            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],initializer=tf.contrib.layers.xavier_initializer())
            #L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
            #L4 = tf.nn.elu(tf.matmul(L3_flat, W4))
            L4 = tf.nn.leaky_relu(tf.matmul(L3_flat, W4),0.1)
            # Dropout Layer4
            L4 = tf.nn.dropout(L4, self.p_keep_hidden)
            
            W_o = tf.get_variable("W_o", shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            
            # Logits layer : Final FC Layer5 Shape = (?, 625) -> 10
            self.logits = tf.matmul(L4, W_o) + b

        # Cost Function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y_onehot))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Test Model
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(Y_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.predict_op = tf.argmax(self.logits, 1, name="pred")

    def train(self, trainData, trainLabel, training = 0.7, hidden = 0.5):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: trainData, self.Y:trainLabel, self.p_keep_conv:training, self.p_keep_hidden:hidden})

    def predict(self, validateData, training = 1.0, hidden = 1.0):
        return self.sess.run(self.logits, feed_dict={self.X : validateData, self.p_keep_conv:training, self.p_keep_hidden:hidden})

    def get_accuracy(self, validateData, validateLabel, training = 1.0, hidden = 1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: validateData, self.Y : validateLabel, self.p_keep_conv:training, self.p_keep_hidden:hidden})



sess = tf.Session()

models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model"+str(m)))
    
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(len(trainData) / batch_size)

    for i in range(total_batch):
        batch_xs = trainData[i*batch_size:(i+1)*batch_size]
        batch_ys = trainLabel[i*batch_size:(i+1)*batch_size].reshape(-1, 1)

        # Train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch: ', '%04d' %(epoch + 1), 'Cost = ', avg_cost_list)
print('Training Finished')


test_size = len(validateLabel)
predictions = np.zeros([test_size, 10])
best_models = []

for m_idx, m in enumerate(models):
    best_models.append(m.get_accuracy(validateData, validateLabel))
    print(m_idx, 'Accuracy: ', best_models[m_idx] )
    p = m.predict(validateData)
    predictions += p
    
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(validateLabel, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

best_model = models[np.argmax(best_models)]

saver = tf.train.Saver()
save_path = saver.save(sess, "/home/itwill03/다운로드/opt2/opt2")
print("Model saved to %s" % save_path)

best_model.logits

        
sess.close()
