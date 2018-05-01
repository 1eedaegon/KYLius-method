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

## 할거
1. 각자 코드 완성/ 여러가지로 돌려보고 정확도랑 인수값 잘 기록해놓기
<pre>
대곤: chroma.stft 3*3/ mfcc(?)
상욱: mfcc 300 등으로 길이 바꿔가면서
승혁: mfcc 하던거 + core.stft
수원: melspectogram / 1d
</pre>

2. 발표할 개요 생각해보기
<pre>
개요
1. data set 설명 (with graph, column, label)
2. sound processing 소개
mfcc, stft, melspectogram 등

코드
3. processing.py 설명
4. mfcc, stft, melspectogram 각각으로 나온 정확도
5. 앙상블로 나온 정확도

마무리
6. 캐글에 올리고, 추천 권유

승혁: sound processing + processing.py
상욱: data set 설명(graph, column, label) mfcc, stft 그래프 등
대곤: 앙상블로 나온 정확도/ 캐글에 올리기
수원: 각각으로 나온 정확도 + 추천 권유
</pre>


## 스케쥴표
<pre>
29일 일요일 오후 2시 사당역.
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
