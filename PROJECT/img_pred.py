import tensorflow as tf
import numpy as np

class img_pred:
    def __init__(self, opt_addr, opt_addr2):
        # initialize/ load
        self.saver=tf.train.import_meta_graph(opt_addr+".meta")
        self.sess = tf.InteractiveSession()
        print("Meta_Graph Imported")
        
        # parameters save 
        self.saver.restore(self.sess, opt_addr)
        print("Parameters Restored")
        
        # variables 
        self.graph=tf.get_default_graph()
        self.X=self.graph.get_tensor_by_name('X:0')
        self.pred=self.graph.get_tensor_by_name('pred:0')
        self.p_keep_conv=self.graph.get_tensor_by_name('p_keep_conv:0')
        self.p_keep_hidden=self.graph.get_tensor_by_name('p_keep_hidden:0')
        print("Variables Saved")
    
    def number(self, imgaddr):
        import matplotlib.pyplot as plt
        from PIL import Image    
        self.imgaddr=imgaddr
        im=Image.open(self.imgaddr)
        img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
        data = img.reshape([1, 784])
        data = 255 - data

        #show image
        plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest')
        plt.show()
        
        #classification result
        self.result=self.sess.run(self.pred, feed_dict={self.X: data, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
        print("MNIST predicted Number : ", self.result)
        
    def file_rename(self):
        import os
        fname=self.imgaddr
        frename=os.path.splitext(fname)[0]+str(self.result)+os.path.splitext(fname)[-1]
        #파일이름 뒤에 분류된 숫자를 넣음
        os.rename(fname, frename)
        print("파일이름이 {}에서 {}로 변경되었습니다".format(fname, frename))