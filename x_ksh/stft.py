def five_sec_extract2(file):
    import librosa
    import numpy as np
    #zero padding to file.shape[0] X 17 X 200    
    array = np.repeat(0., 17 * 200).reshape(17, 200)
    y, sr = librosa.core.load(file, 
                              mono=True, res_type="kaiser_fast")
    stft=librosa.core.stft(y,32,16)
    mag, pha = librosa.magphase(stft)
    length=stft.shape[1]
    abs_mag=np.abs(mag)
    if length == 200:
        array[:, :]=mag
    elif length < 200:
        array[:, :length]=mag
    elif length > 200:
        argmax=np.argmax(abs_mag, axis=1)
        sample=[]
        for i in range(np.max(argmax)):
            sample.append(np.sum((argmax>=i) & (argmax <i+200)))
        start=sample.index(max(sample))
        array=mag[:, start:start+200]
    return(array.reshape(17*200))
