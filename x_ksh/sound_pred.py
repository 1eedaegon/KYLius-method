import tensorflow as tf
import numpy as np

opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh/optx2/optx2"

class sound_pred:
    def __init__(self, opt_addr):
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
    
    def tryit(self, soundaddr):
        
        import librosa
        import numpy as np
        from mfcc_processing import five_sec_extract
        self.soundaddr=soundaddr
        #mfcc processing
        mfcc, _=five_sec_extract(self.soundaddr)
        #classification result
        self.result=self.sess.run(self.pred, feed_dict={self.X: mfcc, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
        print(self.result)
