import numpy as np
idx = np.array(['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
       'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard',
       'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime',
       'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
       'Microwave_oven', 'Burping_or_eructation', 'Double_bass',
       'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano',
       'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
       'Violin_or_fiddle', 'Finger_snapping'])

path='/Users/kimseunghyuck/desktop/'
import os
#folder="audio_test"
file_list=os.listdir(path+'audio_test')
file_list=np.array(file_list).reshape(-1)

#각각의 소프트맥스 데이터를 불러옴.
array1=np.genfromtxt('/Users/kimseunghyuck/desktop/array1.csv', delimiter=',')
array2=np.genfromtxt('/Users/kimseunghyuck/desktop/array2.csv', delimiter=',')
array3=np.genfromtxt('/Users/kimseunghyuck/desktop/array3.csv', delimiter=',')
print(array1.shape, array2.shape, array3.shape)

#가중치를 곱하여 더한 후 argmax
array=0.77/2*array1+0.64*array2+0.76/2*array3
predarray=np.argmax(array, axis=1)

print(file_list.shape, predarray.shape)

import pandas as pd

submission={}
i=0
for file in file_list:
    if (file.split('.')[-1]=="wav"):
        submission[file]=idx[predarray[i]]
    i+=1

#파일 아웃풋
KYLius5=pd.DataFrame([[k,v] for k,v in iter(submission.items())],columns=["fname","label"])
KYLius5.to_csv(path+'KYLius5.csv', header=True, index=False, sep='\t')


