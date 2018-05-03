import tensorflow as tf
import numpy as np

opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT2/optx/optx"

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
        import pandas as pd
        train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
        idx = train.label.unique()        
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        from mfcc import five_sec_extract
        soundaddr='/Users/kimseunghyuck/desktop/audio_train/0a2a5c05.wav'
        testfile8 = pd.read_csv('/Users/kimseunghyuck/desktop/testfile8.csv')
        testfile8 = np.array(testfile8).reshape(-1)
        testLabel8 = np.genfromtxt('/Users/kimseunghyuck/desktop/testLabel8.csv', delimiter=',')
        error=0
        k=0
        errorlist=[]
        for file in testfile8:
            #mfcc processing
            mfcc=five_sec_extract('/Users/kimseunghyuck/desktop/audio_train/'+file)
            #classification result
            result=sess.run(pred, feed_dict={X: mfcc.reshape(1, 20, 430), p_keep_conv: 1.0, p_keep_hidden: 1.0})
            if testLabel8[k] != result:
                print('x : ', file)
                error+=1
                errorlist.append(file)
            else:
                print('o')
            k+=1
        print("error count :", error)
        print("error percentage :", (474-error)/474)

    def close(self):
        sess.close()
