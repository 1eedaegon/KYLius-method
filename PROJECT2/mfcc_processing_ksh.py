#mfcc_processing.py
#librosa.feature.mfcc를 이용해서 mfcc 데이터를 저장하는 코드입니다.

#필요한 모듈 임포트
import librosa
import numpy as np
from matplotlib import pyplot as plt
#import labels
import tensorflow as tf
tf.set_random_seed(777) 
import pandas as pd
train = pd.read_csv('/Users/kimseunghyuck/desktop/sound_train.csv')
#train = pd.read_csv('/home/paperspace/Downloads/audio_train.csv')

#train/test, Data/Label split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size = 0.05)
trainfile = train_set.values[:,0]
testfile = test_set.values[:,0]
trainLabel = train_set.values[:,1]
testLabel = test_set.values[:,1]

#data load and extract mfcc (scaling indluded)
path = '/Users/kimseunghyuck/desktop/'
#path = '/home/paperspace/Downloads/audio_train/'

def see_how_long(file):
    c=[]
    for filename in file:
        y, sr = librosa.core.load(path+'audio_train/'+filename, 
                                  mono=True, res_type="kaiser_fast")
        mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        abs_mfcc=np.abs(mfcc)
        c.append(abs_mfcc.shape[1])
    return(c)
 
#n=see_how_long(trainfile)
#print(np.max(n), np.min(n))      #1292, 14
#n2=see_how_long(testfile)
#print(np.max(n2), np.min(n2))    #1292, 13

#show me approximate wave shape
filename= trainfile[0]
y, sr = librosa.core.load(path+'audio_train/'+filename, 
                          mono=True, res_type="kaiser_fast")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
length=mfcc.shape[1]
plt.plot(mfcc[3,])
plt.plot(np.abs(mfcc[3,]))
plt.plot(mfcc[2,])
plt.plot(np.abs(mfcc[3,]))

#5 seconds(430 segments) extract
def five_sec_extract(file):
    #zero padding to file.shape[0] X 20 X 430
    n=file.shape[0]
    array = np.zeros((n, 20, 430))
    k=0
    see = []
    for filename in file:
        y, sr = librosa.core.load(path+'audio_train/'+filename, 
                                  mono=True, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        length=mfcc.shape[1]
        abs_mfcc=np.abs(mfcc)
        if length == 430:
            array[k, :, :]=mfcc
        elif length < 430:
            tile_num = (430//length)+1
            array[k, :, :]=np.tile(mfcc,tile_num)[:,0:430]
        elif length > 430:
            argmax=np.argmax(abs_mfcc, axis=1)
            sample=[]
            for i in range(np.max(argmax)):
                 sample.append(np.sum((argmax>=i) & (argmax <i+430)))
            start=sample.index(max(sample))
            array[k, :, :]=mfcc[:, start:start+430]
            see.append(start)
        k+=1
    return(array, see)

trainData, see1=five_sec_extract(trainfile)
testData, see2=five_sec_extract(testfile)
print(see1)
print(see2)

print(trainData.shape, testData.shape, trainLabel.shape, testLabel.shape)
# 트레이닝 셋 5%만 뽑음 (8999, 20, 430) (474, 20, 430) (8999,) (474,)

#라벨이 총 몇개가 되어야 하는지 확인
print(len(np.unique(trainLabel)))   #41
print(len(np.unique(testLabel)))    #41

#문자열로 되어있는 라벨을 전부 0~40으로 바꾼다.
def Labeling(label):
    #idx = np.unique(train.values[:,1])     #이건 abc 순
    idx = train.label.unique()
    r=pd.Series(label)
    for i in range(len(idx)):
        r[r.values==idx[i]]=i
    return(r)

trainLabel=Labeling(trainLabel)
testLabel=Labeling(testLabel)
#라벨이 0~40으로 잘 들어갔는지 확인
print(min(trainLabel), max(trainLabel), min(testLabel), max(testLabel))

#트레이닝 및 테스트에 적절히 사용하기 위해 csv파일로 다운로드한다. 
#(3D array는 csv파일로 저장이 안되므로 2D로 변환하여 저장)
trainData2D=trainData.reshape(-1, 20*430)
testData2D=testData.reshape(-1, 20*430)
np.savetxt(path+'trainData8.csv', 
           trainData2D, delimiter=",")
np.savetxt(path+'testData8.csv', 
           testData2D, delimiter=",")
np.savetxt(path+'trainLabel8.csv', 
           trainLabel, delimiter=",")
np.savetxt(path+'testLabel8.csv', 
           testLabel, delimiter=",")
np.savetxt(path+'testfile8.csv', 
           testfile, header = " ", fmt='%s')
np.array(testfile)
testfile.shape

#trainData8 <- mfcc, 20*430, train/test: 95%/5%