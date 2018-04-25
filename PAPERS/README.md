# 읽어볼 논문들

pdf 파일 올린 것 외에 봐볼만한 링크 등 공유

####  Spectrogram, Cepstrum and Mel-Frequency Analysis
http://www.speech.cs.cmu.edu/15-492/slides/03_mfcc.pdf
<pre>
<b>요약</b>
- 짧은 윈도우를 통해 음향을 분석한다. 
각각의 윈도우로부터 하나의 스펙트럼이 얻어진다.(FFT 이용)
- 위에서 얻어진 스펙트럼을 Mel-Filters에 통과시키면 MelSpectrum이 얻어진다.
- Mel-Spectrum을 가지고 'Ceptral 분석'을 수행하면 MFCC (Mel-Frequency Cepstral Coefficients)가 얻어진다.
- 따라서 소리 데이터는 Cepstral 벡터의 시퀀스로 표현할 수 있다.
</pre>


#### MFCC and gammatone filter banks
http://www.cs.tut.fi/~sgn14006/PDF2015/S04-MFCC.pdf

#### Fujitsu, 교토대의대 공동연구 (small image data 로 악성, 약성 판별)
http://www.fujitsu.com/global/about/resources/news/press-releases/2018/0416-01.html#1