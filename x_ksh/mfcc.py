def five_sec_extract(file):
    import librosa
    import numpy as np
    array = np.repeat(0., 20 * 430).reshape(20, 430)
    y, sr = librosa.core.load(file, 
                              mono=True, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    length=mfcc.shape[1]
    if length == 430:
        array=mfcc
    elif length < 430:
        array[:, :length]=mfcc
    elif length > 430:
        sample = np.repeat(0., (length - 430)*20).reshape(20,length - 430)
        for j in range(length - 430):
            for i in range(20):
                sample[i,j]=np.var(mfcc[i,j:j+430])
        A=np.argmax(sample, axis=1)
        start=np.argmax(np.bincount(A))
        array=mfcc[:, start:start+430]
    return(array.reshape(20*430))