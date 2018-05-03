import tensorflow as tf
import numpy as np
path='/Users/kimseunghyuck/desktop/'
opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT2/optx2/optx2"

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
    
    def tryit(self, soundaddr, method):
        import pandas as pd
        train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
        idx = train.label.unique()        
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        if method=="m":
            from mfcc import five_sec_extract
            testfile8 = pd.read_csv('/Users/kimseunghyuck/desktop/testfile8.csv')
            testfile8 = np.array(testfile8).reshape(-1)
            testLabel8 = np.genfromtxt('/Users/kimseunghyuck/desktop/testLabel8.csv', delimiter=',')
            error=0
            k=0
            errorlist1=[]
            for file in testfile8:
                #mfcc processing
                mfcc=five_sec_extract('/Users/kimseunghyuck/desktop/audio_train/'+file)
                #classification result
                result=sess.run(pred, feed_dict={X: mfcc.reshape(1, 20, 430), p_keep_conv: 1.0, p_keep_hidden: 1.0})
                if testLabel8[k] != result:
                    print('x : ', file)
                    error+=1
                    errorlist1.append(file)
                else:
                    print('o')
                k+=1
            print("error count :", error)
            print("error percentage :", (474-error)/474)
            
        elif method=='s':
            from stft import five_sec_extract2
            testfile8 = pd.read_csv('/Users/kimseunghyuck/desktop/testfile8.csv')
            testfile8 = np.array(testfile8).reshape(-1)
            testLabel8 = np.genfromtxt('/Users/kimseunghyuck/desktop/testLabel8.csv', delimiter=',')
            error=0
            k=0
            errorlist2=[]
            for file in testfile8:
                #mfcc processing
                stft=five_sec_extract2('/Users/kimseunghyuck/desktop/audio_train/'+file)
                #classification result
                result=sess.run(pred, feed_dict={X: stft.reshape(1, 17, 200), p_keep_conv: 1.0, p_keep_hidden: 1.0})
                if testLabel8[k] != result:
                    print('x : ', file)
                    error+=1
                    errorlist2.append(file)
                else:
                    print('o')
                k+=1
            print("error count :", error)
            print("error percentage :", (474-error)/474)    

    def submission(self, folder):
        import pandas as pd
        train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
        idx = train.label.unique()        
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        file_list=os.listdir(path+"audio_test")
        file_list=np.array(file_list).reshape(-1)
        submission={}
        
            from stft import five_sec_extract2
            for file in file_list:
                if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                    #mfcc processing
                    stft=five_sec_extract2('/Users/kimseunghyuck/desktop/audio_test/'+file)
                    #classification result
                    result=sess.run(pred, feed_dict={X: stft.reshape(1, 17, 200), p_keep_conv: 1.0, p_keep_hidden: 1.0})
                    submission[file]=idx[result[0]]

len(submission)
submission['0b0427e2.wav']=idx[0]
submission['6ea0099f.wav']=idx[0]
submission['b39975f5.wav']=idx[0]
len(submission)
KYLius2=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["fname","label"])
KYLius2.to_csv(path+'KYLius2.csv', header=True, index=False, sep='\t')


    def close(self):
        sess.close()

np.savetxt(path+'errorlist1.csv', 
           errorlist1, header = " ", fmt='%s')
        
np.savetxt(path+'errorlist2.csv', 
           errorlist2, header = " ", fmt='%s')

errorlist1 = pd.read_csv('/Users/kimseunghyuck/desktop/errorlist1.csv')
errorlist2 = pd.read_csv('/Users/kimseunghyuck/desktop/errorlist2.csv')


type(errorlist1)
type(errorlist2)

#교집합은 어떻게 되는가?
intersect = [x for x in errorlist1.values.tolist() if x in errorlist2.values.tolist()]
len([x for x in errorlist1.values.tolist() if x in errorlist2.values.tolist()])
#49
len(errorlist1.values.tolist())     #136
len(errorlist2.values.tolist())     #101

