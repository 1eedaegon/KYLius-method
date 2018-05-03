import numpy as np
def predict(array1, array2, array3):
    idx = np.array(['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
       'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard',
       'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime',
       'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
       'Microwave_oven', 'Burping_or_eructation', 'Double_bass',
       'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano',
       'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
       'Violin_or_fiddle', 'Finger_snapping'])
    array=array1+array2+array3
    predarray=np.argmax(array, axis=1)

array1=np.genfromtxt('/Users/kimseunghyuck/desktop/array1.csv', delimiter=',')
array2=np.genfromtxt('/Users/kimseunghyuck/desktop/array2.csv', delimiter=',')
array3=np.genfromtxt('/Users/kimseunghyuck/desktop/array3.csv', delimiter=',')


print(array1.shape, array2.shape, array3.shape)


array=0.77*array1+0.64*array2+0.76*array3

predarray=np.argmax(array, axis=1)

path='/Users/kimseunghyuck/desktop/'
import os
#folder="audio_test"
file_list=os.listdir(path+'audio_test')
file_list=np.array(file_list).reshape(-1)






