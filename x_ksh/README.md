# Smallerhand; 김승혁 (KYLius-method)

<b>진행상황</b>
<pre>
아래로 갈수록 과거에 돌린 모델 (맨 위에 있는 게 현재 돌리고 있는 모델)
</pre>

## 현재 시도중인 모델
<pre>
1. core.stft

2. mfcc 를 데이터 분산값 크게 들어가게 해서 20*430으로 자름.

3. 이미 한거 : mfcc 최대값 많이 들어가는 걸로 20*430으로 잘라서 processing,
트레이닝 모델은 크게 다르지 않음.
</pre>

## 완성 모델 1 - 67~74%(에폭 600~700 사이에서 67~74 왔다갔다 함)
<pre>
preparing_data.py - 데이터 전처리
SA_cnn_ps3.py - 모델
아래 설명은 다시 수정해야 함.
</pre>

preparing_data.py
<pre>
이 코드로 trainData.csv, trainLabel.csv, testData.csv, testLabel.csv 를 만들어 저장함. (전체 약 600MB)
sklearn 모듈 이용, 7:3으로 train/test split.
see_how_long 은 mfcc 뽑았을 때 파일별로 길이가 어느 정도인지 확인한 것. (이후 안쓰이니 신경 안써도 됨)

n_mfcc=20으로 mfcc로 변환한 후, (길이가 26~2584로 들쭉날쭉한데) 100으로 통일함.
어떻게 했냐면,
(short_time_extrat 함수)
- 길이가 (드물겠지만) 100이면 그대로 array에 넣음.
- 길이가 100보다 작으면 나머지는 0으로 채운 뒤 array에 넣음.
- 길이가 100보다 크면 (이 부분이 약간 복잡한데),
20개 특성 주파수 각각 중 절대값의 최대값을 가지고 있는 구간(np.argmax 이용)을 뽑아서,
i~i+100 구간을 정할 때 최대값이 가장 많이 포함되는 i 값 (0< i < 데이터 길이) 을 찾은 뒤,
i~i+100 까지만 잘라서 넣음.
<code>
            argmax=np.argmax(abs_mfcc, axis=1)
            sample=[]
            for i in range(np.max(argmax)):
                 sample.append(np.sum((argmax>=i) & (argmax <i+100)))
            start=sample.index(max(sample))
            array[k, :, :100]=abs_mfcc[:, start:start+100]
</code>

데이터 라벨링은 0~40의 숫자로 치환함.
(결과물 낼 때는 이부분에서 영어: 숫자 저장한 csv도 있어야 할 것임. 여기서는 생략)

이렇게 해서 (추후 모델 돌릴 때 편의를 위해) 저장함.
</pre>

SA_cnn_ps3.py
<pre>
preparing_data.py 로 저장시킨 데이터 일단 불러온 뒤 reshape.
shape이 맞는지 확인.
conv2d L1 : 20*100*1 로 입력 받아서 10*20*32 로 방출
    (윈도우 사이즈 =  2 * 10, 맥스풀 사이즈 = 2 *5)
conv2d L2 : 10*20*32 로 받아서 4*7*64 로 방출
    (win = 2 * 2, max_pool = 3 * 3)
conv2d L3 : 4* 7* 64 로 받아서 2*3*128 로 (flat하게) 방출
    (win = 2 * 2, max_pool = 3 * 3)
FC L4: 2*3*64 로 받아서 41 로 방출
    (2*3*128 -> 615 -> 41)

정확도: 51~60%

윈도우 사이즈 등 바꿔서 계속 돌려봄
현재까지 아래처럼 구성했을 때 정확도 가장 높음.
win : (2, 10), (2,4), (2,3)
max_pool : (2,5), (3,3), (3,3)
accuracy: 53~65%
</pre>

## 일단 보류 중인 모델 (코드 짜는 데 오래 걸릴 것 같아서 시간상 보류)
<pre>
코드 아직 못 짬
</pre>
list안에 array를 담아서 가변적인 길이의 mfcc output을 담은 다음에 CNN모델(아마 파이썬으로 코드 짜야 할 듯)에 넣고,
<br>
20*20(n_mfcc를 20으로 한다고 가정할 때) 의 window 로 지나가면서 output을 배출.
<br>
그 중에 max값을 뽑아낸다. 아주 간단한 구조지만 텐서플로우에 제공되는 함수가 없으므로 파이썬으로 짜야 할 것임.
<br>
짜는 게 좀 어려울 수 있고, 원만히 돌아간다면 보다 정교화할 계획.
<pre>
장점1: 효율적임.
장점2: 필요한 정보만 얻을 수 있다.
장점3: 연속적인 흐름(파형)을 파악할 수 있다.
장점4: 시간이 앞의 모델보다 훨씬 적게 걸릴 것이다.
단점: 구현하기 어렵다.
</pre>

## 지난 모델 2 (돌아가긴 함)
<pre>
sound_analysis5_cnn_ps.py
</pre>
0으로 전부 패딩(전처리과정) 해서 길이를 같게 만든 다음 CNN -> Fully Connectied -> Softmax
<pre>
정확도: 에폭 50에서 20%대. (낮음)
시간: 오래걸림
가능성: 아주 간단하게 conv2d하나 + FC 하나만 썼는데 20% 대이므로 정교하게 구성하면 높아질 수 있을 것 같음.
단점1: 데이터 크기가 너무 커짐. (csv로 다운받으니 전체 15GB 정도 됨)
단점2: 그래서 시간이 오래걸리고 메모리 관리가 힘듬
단점3: 같은 얘기지만 비효율적임.
</pre>

## 지난 모델 1 (돌아가긴 함)
<pre>
sound_analysis2.py
sound_analysis3.py
</pre>
<code>
import tensorflow as tf
</code>
기준으로 앞부분은 데이터 디스플레이만 해봄.
그 뒷부분만 보면 됨.

기본 구조는

1. 데이터 전처리 1
<pre>
y, sr = soundfile.read(path+filename, dtype='float32')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
max_mfcc=np.max(mfcc, axis=1)
mins, maxs=np.min(max_mfcc), np.max(max_mfcc)
scaled_mfcc=(max_mfcc-mins)/(maxs-mins)
dic[i] = scaled_mfcc
soundfile.read 로 y, sr 추출
librosa.feature.mfcc 로 mfcc 변환 (n_mfcc=20)
20개 속성들 각각의 max값을 뽑음
전체를 min, max scaling
-> 이렇게 해서 하나의 sound 파일 당 20개의 값이 얻어짐.
</pre>


2. 데이터 전처리 2
<pre>
라벨은 0~40의 숫자로 바꿈
</pre>


3. MLP 설계
<pre>
특별한 의미는 없이 히든레이어 하나, 인풋, 아웃풋도 각각 하나로 해서 3개 레이어 만듬.
(처음엔 히든 2개인 총 4개 레이어로 해봤는데 돌릴수록 정확도가 떨어지는게 오버피팅 되는 것 같아서 3개로 줄임)
첫번째 층: 20 -> 128
두번째 층: 128 -> 256
세번째 층: 256 -> 41
소프트맥스, 아담으로 돌림.
</pre>


4. 돌려봄
<pre>
1)
lr=0.001
epoch = 1000
keep_prob = 1
정확도 32.4~32.9%

2)
다른 조건 같음
keep_prob = 0.8 (트레이닝때만)
정확도 36.9~37.9%
최고 39.0%

특이사항: lr을 저거보다 높게 하거나 낮게 하면 정확도가 확 떨어짐.
</pre>


5. 개선할 부분
<pre>
파라미터를 좀 바꾸면 좀더 높아질지도 모르지만,
음색을 구분하려면, 그리고 소리에 담긴 뉘앙스를 구분하려면 아무래도
시간적으로 연속적인 파형에서 단편적인 속성값만 뽑아내서는 안되고 연속적인 인식이 가능한
CNN이나 RNN, LSTM을 써야 할 것이라고 생각함.
</pre>


