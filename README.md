# KYLius-method
<p><i>2번째 도전</i></p>
<img src="PROJECT1/logo/KYLius_logo4.png">

## Second Challenge
<p>

### Freesound General-Purpose Audio Tagging Challenge
<br>
Can you automatically recognize sounds from a wide range of real-world environments?
<br>
https://www.kaggle.com/c/freesound-audio-tagging
<br>
기간: 4월 23일 (월요일) ~ 5월 3일 (목요일) 약 2주간
</p>
<br>

## 폴더 설명
##### 우리 KYLius 의 깃허브 repository 는 세 가지 공간으로 나눠서 관리하고 있습니다(아래).
1. 각자의 개인 폴더 <br>
<pre> 각자 개인적인 작업, 또는 팀프로젝트 관련 작업이지만 아직 공유하기엔 정리가 덜 된 것들을 모아놓는 공간입니다. <br>
개인적인 공간이긴 하지만 누가 보더라도 대략 뭘 하고 있는지 알아볼 수 있게끔 적절한 주석은 필수!<br>
깔끔한 정렬을 위해 앞에 x를 붙이고, x뒤에 이니셜을 붙임. </pre>
2. PAPERS - 논문 및 공부할 자료 <br>
<pre> 우리가 함께 공부하면 좋을 논문이나 자료들을 올리는 공간입니다. </pre>
3. PROJECT - 치열한 공개의 장 <br>
<pre> 엄선된 자료, 혹은 모듈식 개발에 끼워넣을 수 있는 완성된 코드만 올리는 공간입니다. </pre>
<br>

##### 4월 25일 회의
<pre>
대곤: 전처리하는 방법(과정). 샘플링(아날로그-> 디지털),
중간에 짤린걸 라이키스트 정리로 사이값 보정.
fft 로 분류. mfcc(샘플링, 라이키스트, fft까지 다 됨) 에 이런 기능이 있음.
</pre>

<pre>
수원: 전처리/ conv2d로 돌린거 깃에 올림(실패)
mfcc같은 값들이 몇가지 종류가 있음.
본 사이트 중에 mfcc에다 다른 특징값을 붙여넣는 식으로 전처리.
mfcc로 하면 41개 속성값이 나옴. 다른 특징 붙이니까 192개로 늘어남.
</pre>

<pre>
상욱: librosa 로 전처리 다 했음. mfcc만 해봄.
조대협 블로그에서 mfcc 행을 25개 주라고 함.
그렇게 했는데 안돼서 캐글 커널처럼 40개로 함.
원핫 썼는데 라벨이 안읽혀서 테스트 못해보고 있음.
</pre>

<pre>
승혁: librosa feature 뽑아내봄
</pre>

<pre>
할거:
1. 음향학 공부(파이썬 함수 각각의 feature 중 뭘 써야 할지, 왜 쓰는지) + 전처리를 어떻게 할지
- 대곤(librosa, mfcc) <- librosa로 뽑아낼 수 있는 특징값 중에 mfcc는 다 뽑히는데 테스트셋에서 mfcc외의 특징은 잘 안뽑히는데 알아봐주세요.(수원 요청)
2. conv1d - 승혁, 상욱
3. conv2d - 수원, 상욱
</pre>

## 스케쥴표
<pre>
25~27일 금요일 맡은거 해보고 오기.
27일 금요일 5시 회의.
29일 일요일 오후 회의.
5월 2일부터 피피티.
5월 4일 발표.
</pre>

## 추가 정보
연철이가 test.csv 중에 안되는 파일 목록 알려줌(3개)
<pre>
0b0427e2, 6ea0099f, b39975f5
</pre>

## 지난 주제: 
1. MNIST DIGITS RECOGNIZER (2018. 4/9 ~ 4/20)
<pre>
캐글 MNIST 데이터셋을 가지고 99.5% 정확도로 숫자 이미지를 분류하는 모델, 프로그램을 만듬(Tensorflow, CNN 활용). <br>
데이터셋: https://www.kaggle.com/c/digit-recognizer
</pre>
