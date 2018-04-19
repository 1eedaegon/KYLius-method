import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import sys
import matplotlib.pyplot as plt
%matplotlib inline

#데이터 가져오기
#train = pd.read_csv('c:/python/train.csv')
train = pd.read_csv('/home/itwill03/다운로드/train.csv')

#훈련세트, validation세트 나누기(여기서는 validate만 필요)
from sklearn.model_selection import train_test_split
train_set, validate_set = train_test_split(train, test_size = 0.3)
#trainData = train_set.values[:,1:]
validateData1 = validate_set.values[:,1:].astype(float)
#trainLabel=train_set.values[:,0]
validateLabel1 = validate_set.values[:,0]


#saver.save(sess, '/home/itwill03/다운로드/cnn_session')
#save_path = saver.save(sess, "/home/itwill03/다운로드//opt2")

# initialize
saver = tf.train.import_meta_graph("/home/itwill03/다운로드/opt2.meta")
#saver = tf.train.import_meta_graph("/home/itwill03/다운로드/cnn_session.meta")
sess = tf.InteractiveSession()
saver.restore(sess, tf.train.get_checkpoint_state("/home/itwill03/다운로드/opt2/").model_checkpoint_path)
print("Parameters Restored")

graph=tf.get_default_graph()
X=graph.get_tensor_by_name('X:0')
pred=graph.get_tensor_by_name('pred:0')
p_keep_conv=graph.get_tensor_by_name('p_keep_conv:0')
p_keep_hidden=graph.get_tensor_by_name('p_keep_hidden:0')

# test the data
print(sess.run(pred, feed_dict={X: validateData1, 
                p_keep_conv: 1.0, p_keep_hidden: 1.0}))

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

    result = sess.run(pred, feed_dict={X:data, training:False})
    result_show.append(sess.run(tf.argmax(result, 1)))
print("MNIST predicted Number")
print(result_show)  

# one image test 

im=Image.open("/home/itwill03/다운로드/numbers_image/number5.jpeg")
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
data = img.reshape([1, 784])
data = 1-(data/255)
plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest')
result = sess.run(pred, feed_dict={X:data, training:False})
print("MNIST predicted Number : ", sess.run(tf.argmax(result, 1)))

sess.close()

