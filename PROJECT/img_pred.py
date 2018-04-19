import tensorflow as tf
import numpy as np

class img_pred:
    def __init__(self, opt_addr, opt_addr2):
        # initialize/ load
        """
        import sys
        sys.path.append("/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT")
        sys.path
        opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT/opt3/opt3"
        opt_addr2="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT/opt3"
        """
        self.saver=tf.train.import_meta_graph(opt_addr+".meta")
        self.sess = tf.InteractiveSession()
        print("Meta_Graph Imported")
        
        #saver.restore(sess, tf.train.get_checkpoint_state(opt_addr2).model_checkpoint_path)
        self.saver.restore(self.sess, opt_addr)
        print("Parameters Restored")
        
        self.graph=tf.get_default_graph()
        self.X=self.graph.get_tensor_by_name('X:0')
        self.pred=self.graph.get_tensor_by_name('pred:0')
        self.p_keep_conv=self.graph.get_tensor_by_name('p_keep_conv:0')
        self.p_keep_hidden=self.graph.get_tensor_by_name('p_keep_hidden:0')
        print("Variables Saved")
    
    def number(self, arg1):
        import matplotlib.pyplot as plt
        from PIL import Image    
        im=Image.open(arg1)
        img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
        data = img.reshape([1, 784])
        data = 255 - data
        plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest')
        print("MNIST predicted Number : ",self.sess.run(self.pred, feed_dict={self.X: data, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0}))
        
    
