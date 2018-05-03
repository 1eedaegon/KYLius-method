import tensorflow as tf
import numpy as np

opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh/optx2/optx2"

class sound_pred:
    def __init__(self, opt_addr):
        # initialize/ load
        saver=tf.train.import_meta_graph(opt_addr+".meta")
        sess = tf.InteractiveSession()
        print("Meta_Graph Imported")
        
        # parameters save 
        saver.restore(sess, opt_addr)
        print("Parameters Restored")
        
        # variables 
        graph=tf.get_default_graph()
        X=graph.get_tensor_by_name('X:0')
        pred=graph.get_tensor_by_name('pred:0')
        p_keep_conv=graph.get_tensor_by_name('p_keep_conv:0')
        p_keep_hidden=graph.get_tensor_by_name('p_keep_hidden:0')
        print("Variables Saved")
    
    def tryit(self, soundaddr):
        
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')
        
        from stft import five_sec_extract2
        soundaddr='/Users/kimseunghyuck/desktop/audio_train/0a0a8d4c.wav'
        #mfcc processing
        stft=five_sec_extract(soundaddr)
        #classification result
        result=sess.run(pred, feed_dict={X: stft.reshape(17, 200), p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print(self.result)
sess.close()