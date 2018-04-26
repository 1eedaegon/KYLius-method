# Smallerhand; 김승혁 (KYLius-method)
<p><i>내가 생각하는 주제...</i></p>
<pre>
<b>진행상황</b>
sound_analysis2.py - 이거 하나만 보면 됨.
<code>
import tensorflow as tf
</code>
기준으로 앞부분은 데이터 디스플레이만 해봄.
그 뒷부분만 보면 됨.

기본 구조는
1. 데이터 전처리 1
<code>
y, sr = soundfile.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
max_mfcc=np.max(mfcc, axis=1)
mins, maxs=np.min(max_mfcc), np.max(max_mfcc)
scaled_mfcc=(max_mfcc-mins)/(maxs-mins)
dic[i] = scaled_mfcc
</code>
soundfile.read 로 y, sr 추출
librosa.feature.mfcc 로 mfcc 변환 (n_mfcc=20)
20개 속성들 각각의 max값을 뽑음
전체를 min, max scaling
-> 이렇게 해서 하나의 sound 파일 당 20개의 값이 얻어짐.

2. 데이터 전처리 2
라벨은 0~40의 숫자로 바꿈

3. MLP 설계
특별한 의미는 없이 히든레이어 하나, 인풋, 아웃풋도 각각 하나로 해서 3개 레이어 만듬.
(처음엔 히든 2개인 총 4개 레이어로 해봤는데 돌릴수록 정확도가 떨어지는게 오버피팅 되는 것 같아서 3개로 줄임)
첫번째 층: 20 -> 128
두번째 층: 128 -> 256
세번째 층: 256 -> 41
소프트맥스, 아담으로 돌림.

4. 돌려봄
1)
lr=0.001
epoch = 1000
keep_prob = 1
정확도 32.4~32.9%

2)
다른 조건 같음
keep_prob = 0.8 (트레이닝때만)
정확도 36.9~37.9%

5. 개선할 부분
파라미터를 좀 바꾸면 좀더 높아질지도 모르지만,
음색을 구분하려면, 그리고 소리에 담긴 뉘앙스를 구분하려면 아무래도
시간적으로 연속적인 파형에서 단편적인 속성값만 뽑아내서는 안되고 연속적인 인식이 가능한
CNN이나 RNN, LSTM을 써야 할 것이라고 생각함.
</pre>


