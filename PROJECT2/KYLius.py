import tensorflow as tf
import numpy as np
import pandas as pd

#opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT2/mfcc/opt"
#opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT2/stft/opt"
#opt_addr="/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/PROJECT2/one/optx"

class sound_pred:
    def __init__(self, opt_addr):
        # initialize/ load
        self.idx = np.array(['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
       'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard',
       'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime',
       'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
       'Microwave_oven', 'Burping_or_eructation', 'Double_bass',
       'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano',
       'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
       'Violin_or_fiddle', 'Finger_snapping'])
    
        self.path='/Users/kimseunghyuck/desktop/'
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
        self.softmax=self.graph.get_tensor_by_name('softmax:0')
        print("Variables Saved")
    
    def tryit(self, soundaddr, method):
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        if method=="m":
            from mfcc import five_sec_extract
            #testfile8 = pd.read_csv('/Users/kimseunghyuck/desktop/testfile8.csv')
            test = pd.read_csv(self.path+soundaddr)
            test = np.array(test).reshape(-1)
            testLabel = np.genfromtxt(self.path+'testLabel8.csv', delimiter=',')
            error=0
            k=0
            errorlist1=[]
            for file in test:
                #mfcc processing
                mfcc=five_sec_extract(self.path+'audio_train/'+file)
                #classification result
                result=self.sess.run(self.pred, feed_dict={self.X: mfcc.reshape(1, 20, 430), 
                                                      self.p_keep_conv: 1.0, 
                                                      self.p_keep_hidden: 1.0})
                if testLabel[k] != result:
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
            #testfile8 = pd.read_csv('/Users/kimseunghyuck/desktop/testfile8.csv')
            test = pd.read_csv(self.path+soundaddr)
            test = np.array(test).reshape(-1)
            testLabel = np.genfromtxt(self.path+'testLabel8.csv', delimiter=',')
            error=0
            k=0
            errorlist2=[]
            for file in test:
                #mfcc processing
                stft=five_sec_extract2(self.path+'audio_train/'+file)
                #classification result
                result=self.sess.run(self.pred, feed_dict={self.X: stft.reshape(1, 17, 200), 
                                                 self.p_keep_conv: 1.0, 
                                                 self.p_keep_hidden: 1.0})
                if testLabel[k] != result:
                    print('x : ', file)
                    error+=1
                    errorlist2.append(file)
                else:
                    print('o')
                k+=1
            print("error count :", error)
            print("error percentage :", (474-error)/474)    

    def submission1(self, folder):  
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        #folder="audio_test"
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        submission={}
        from mfcc import five_sec_extract
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #mfcc processing
                mfcc=five_sec_extract(self.path+'audio_test/'+file)
                #classification result
                result=self.sess.run(self.pred, feed_dict={self.X: mfcc.reshape(1, 20, 430), 
                                                      self.p_keep_conv: 1.0, 
                                                      self.p_keep_hidden: 1.0})
                submission[file]=self.idx[result[0]]
        #len(submission)
        submission['0b0427e2.wav']=self.idx[0]
        submission['6ea0099f.wav']=self.idx[0]
        submission['b39975f5.wav']=self.idx[0]
        #len(submission)
        KYLius1=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["fname","label"])
        KYLius1.to_csv(self.path+'KYLius1.csv', header=True, index=False, sep='\t')

    def submission2(self, folder):  
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        submission={}
        from stft import five_sec_extract2
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #mfcc processing
                stft=five_sec_extract2(self.path+'audio_test/'+file)
                #classification result
                result=self.sess.run(self.pred, feed_dict={self.X: stft.reshape(1, 17, 200), 
                                                      self.p_keep_conv: 1.0, 
                                                      self.p_keep_hidden: 1.0})
                submission[file]=self.idx[result[0]]

        #len(submission)
        submission['0b0427e2.wav']=self.idx[0]
        submission['6ea0099f.wav']=self.idx[0]
        submission['b39975f5.wav']=self.idx[0]
        #len(submission)
        KYLius2=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["fname","label"])
        KYLius2.to_csv(self.path+'KYLius2.csv', header=True, index=False, sep='\t')

    def submission3(self, folder):  
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        #folder="audio_test"
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        submission={}
        from mfcc_only1 import five_sec_extract3
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #mfcc processing
                mfcc=five_sec_extract3(self.path+'audio_test/'+file)
                #classification result
                result=self.sess.run(self.pred, feed_dict={self.X: mfcc.reshape(1, 40, 350, 1), 
                                                      self.p_keep_conv: 1.0, 
                                                      self.p_keep_hidden: 1.0})
                submission[file]=self.idx[result[0]]
        #len(submission)
        submission['0b0427e2.wav']=self.idx[0]
        submission['6ea0099f.wav']=self.idx[0]
        submission['b39975f5.wav']=self.idx[0]
        #len(submission)
        KYLius3=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["fname","label"])
        KYLius3.to_csv(self.path+'KYLius3.csv', header=True, index=False, sep='\t')


    def close(self):
        self.sess.close()

    def softmax1(self, folder):  
        #mfcc 1
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        #folder="audio_test"
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        length = len(file_list)
        softmax = np.zeros((length, 41))
        from mfcc import five_sec_extract
        k=0
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #mfcc processing
                mfcc=five_sec_extract(self.path+'audio_test/'+file)
                #classification result
                softmax[k,] = self.sess.run(self.softmax, 
                       feed_dict={self.X: mfcc.reshape(1, 20, 430), 
                                  self.p_keep_conv: 1.0, 
                                  self.p_keep_hidden: 1.0})
            k+=1
        return(softmax)
    
    def softmax2(self, folder):  
        # stft
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        #folder="audio_test"
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        length = len(file_list)
        softmax = np.zeros((length, 41))
        from stft import five_sec_extract2
        k=0
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #stft processing
                stft=five_sec_extract2(self.path+'audio_test/'+file)
                #classification result
                softmax[k,] = self.sess.run(self.softmax, 
                       feed_dict={self.X: stft.reshape(1, 17, 200), 
                                  self.p_keep_conv: 1.0, 
                                  self.p_keep_hidden: 1.0})
            k+=1
        return(softmax)
    
    def softmax3(self, folder):  
        # mfcc only label 1
        import sys
        sys.path.append('/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/x_ksh')        
        import os
        #folder="audio_test"
        file_list=os.listdir(self.path+folder)
        file_list=np.array(file_list).reshape(-1)
        length = len(file_list)
        softmax = np.zeros((length, 41))
        from label1 import five_sec_extract3
        k=0
        for file in file_list:
            if (file.split('.')[-1]=="wav") & (file not in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']):
                #mfcc processing
                mfcc=five_sec_extract3(self.path+'audio_test/'+file)
                #Normalization
                mean = np.mean(mfcc, axis=0)
                std = np.std(mfcc, axis=0)
                mfcc = (mfcc - mean)/std
                #classification result
                softmax[k,] = self.sess.run(self.softmax, 
                       feed_dict={self.X: mfcc.reshape(1, 40, 350), 
                                  self.p_keep_conv: 1.0, 
                                  self.p_keep_hidden: 1.0})
            k+=1
        return(softmax)
    

"""
np.savetxt(path+'errorlist1.csv', 
           errorlist1, header = " ", fmt='%s')
        
np.savetxt(path+'errorlist2.csv', 
           errorlist2, header = " ", fmt='%s')

errorlist1 = pd.read_csv('/Users/kimseunghyuck/desktop/errorlist1.csv')
errorlist2 = pd.read_csv('/Users/kimseunghyuck/desktop/errorlist2.csv')

#교집합은 어떻게 되는가?
intersect = [x for x in errorlist1.values.tolist() if x in errorlist2.values.tolist()]
len([x for x in errorlist1.values.tolist() if x in errorlist2.values.tolist()])
#49
len(errorlist1.values.tolist())     #136
len(errorlist2.values.tolist())     #101
"""