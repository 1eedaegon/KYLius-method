# Freesound General-Purpose Audio Tagging Challenge
## KYLius의 2번째 프로젝트 페이지에 오신 것을 환영합니다.
#### Can you automatically recognize sounds from a wide range of real-world environments?
https://www.kaggle.com/c/freesound-audio-tagging


### 데이터 전처리 코드
<pre>
extract_features.py - fixed mean features 뽑아내는 코드
mel_extract_128.py - mel spectogram 을 제로패딩만 해서 뽑아내는 코드
mfcc_processing_ksh.py - mfcc 데이터를 정해진 길이로 뽑아내는 코드
stft_processing_ksh.py - stft 데이터를 정해진 길이로 뽑아내는 코드
</pre>

### 트레이닝 코드
<pre>
audio_conv1d_features.py - mixed mean features 데이터로 cnn1d 학습
audio_conv2d_features.py - mixed mean features 데이터로 cnn2d 학습
audio_conv2d_mel.py - mel spectogram 데이터로 cnn2d 학습
mfcc_cnn_ksh.py - mfcc 데이터로 cnn 학습 시키는 코드
stft_cnn_ksh.py - stft 데이터로 cnn 학습 시키는 코드
sound_cnn(ksw).py
</pre>

### 학습된 모델 로드, 오답노트 작성, 캐글에 올릴 데이터 출력, 소프트맥스 값 저장
<pre>
KYLius.py
</pre>

### 소프트맥스 값을 합쳐서 최종 결과 저장
<pre>
RealPred.py
</pre>
