def five_sec_extract3(file):
    import librosa
    import numpy as np
    #zero padding to file.shape[0] X 20 X 430
    array = np.zeros((40, 350))
    y, sr = librosa.core.load(file, 
                              mono=True, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    length=mfcc.shape[1]
    abs_mfcc=np.abs(mfcc)
    if length == 350:
        array=mfcc
    elif length < 350:
        tile_num = (350//length)+1
        array=np.tile(mfcc,tile_num)[:,0:350]
    elif length > 350:
        argmax=np.argmax(abs_mfcc, axis=1)
        sample=[]
        for i in range(np.max(argmax)):
            sample.append(np.sum((argmax>=i) & (argmax <i+350)))
        start=sample.index(max(sample))
        array=mfcc[:, start:start+350]
    return(array.reshape(40*350))