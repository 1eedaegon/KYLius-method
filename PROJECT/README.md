# KYLius DIGITS IMAGE RECOGNITION
## 숫자 이미지 인식 프로그램 만들기

<pre>
숫자 이미지를 input 하면 바로 어느 숫자인지를 분류해주고,
추가적으로 이미지 파일 이름에 라벨을 달아주는 프로그램을 만들어보겠습니다.
(CNN 알고리즘 이용)
</pre>

### 1. train_optimizer.py 를 실행시켜 모델을 트레이닝시키고, 저장.

### 2. img_pred.py 로 저장된 모델을 읽어들인다.
<p>
from img_pred import img_pred <br>
A=img_pred("opt3/opt3", "opt3")
</p>

### 3. number(imgaddr) 함수로 이미지가 어느 숫자인지 출력.
<p>
A.number("numbers_set2/number_a.jpeg")
</p>

### 4. file_rename() 함수로 파일 이름을 변경.
<p>
A.file_rename()
</p>

### ++ image_print.py 로 csv 파일을 이미지로 출력해볼 수 있습니다.
