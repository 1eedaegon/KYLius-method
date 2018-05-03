def five_sec_extract(file):
    import librosa
    import numpy as np
    #zero padding to file.shape[0] X 20 X 430
    array = np.zeros((20, 430))
    y, sr = librosa.core.load(file, 
                              mono=True, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    length=mfcc.shape[1]
    abs_mfcc=np.abs(mfcc)
    if length == 430:
        array=mfcc
    elif length < 430:
        tile_num = (430//length)+1
        array=np.tile(mfcc,tile_num)[:,0:430]
    elif length > 430:
        argmax=np.argmax(abs_mfcc, axis=1)
        sample=[]
        for i in range(np.max(argmax)):
            sample.append(np.sum((argmax>=i) & (argmax <i+430)))
        start=sample.index(max(sample))
        array=mfcc[:, start:start+430]
    return(array.reshape(20*430))
