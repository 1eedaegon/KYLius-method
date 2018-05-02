## 1eedaegon - 이대곤(KYLius-method)

### MFCC(Mel-frequency ceptral coefficients)
<pre>
1. 시간별 소리신호를 작은 크기의 프레임으로 자른다. sampling
2. 각 프레임에 대해서 FTT적용, 주파수 계산 
3. 사람이 인식하는 소리영역으로 자름(15~15000hz), 주파수에서 mel scale로 간격나눔, 반대 공식도 있다.(MF)
4. 로그함수취함(값이 클수록 사람이 듣는 소리가 비례해서 커지는건 아님)
5. DCT == 필터뱅크 각각의 소리에 대해 행렬로 바꾼다.
6. 가장 특징이 두드러지는 소리값을 뽑는다.- Ceptral-Coefficient(CC)
</pre>
