# SPPNET Prior Learning

# IOU(Intersection over Union)

- Intersection over Union은 **object detector의 정확도를 측정하는데 이용되는 평가 지표**

→ Object detection 논문이나 대회에서 IoU 평가 지표를 쉽게 볼 수 있음

- IOU를 적용하기 위해서는 두가지가 필요
1. **ground-truth bounding boxes**(testing set에서 object 위치를 labeling 한것)
2. **prediceted bounding boxes** (model이 출력한 object 위치 예측값)

![Untitled](SPPNET%20Pri%205e4b0/Untitled.png)

- predicted bounding box는 빨강색이고, ground-truth는 초록색

- object detector가 얼마나 정확히 객체 위치를 탐지했는지 알아보기 위해 IoU를 계산

![Untitled](SPPNET%20Pri%205e4b0/Untitled%201.png)

→ IOU를 계산하는 식

- **area of overlab**은 prediceted bounding box와 ground-truth bounding box가 겹치는 부분
- **area of unin**은 predicted bounding box와 ground-truth bounding box를 둘러싸는 영역

### 왜 IOU를 사용하는가?

- 머신 러닝으로 분류 문제를 수행할 때 predicted class가 맞는지 틀렸는지 쉽게 확인할 수 있음

→ 맞다 틀리다 두 가지로 구분

- **BUT,**  object detection에서는 간단하지 않음

→ 현실 문제에서 predicted bounding box가 정확히 ground-truth bounding box와 일치하지 경우는 거의 없기 때문

→ model은 다양한 parameters(image pyramid scale, sliding window size, feature extraction method 등등)가 있기 때문

⇒ predicted bounding box가 ground-truth bounding box와 **얼마나 일치하는지 측정하기 위한 평가 지표**를 정의

![Untitled](SPPNET%20Pri%205e4b0/Untitled%202.png)

- IoU가 높은 경우 predicted bounding box와 ground-truth bounding box가 거의 포개진 것을 확인할 수 있음

# NMS(Non-maximum Suppression)

- non-max suppresion은 object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법

- 이미지에서 객체는 다양한 크기와 형태로 존재
- 이것을 완벽하게 검출하기 위해 object detection 알고리즘은 여러 개의 bounding boxes를 생성

⇒ **하나의 bounding box만을 선택해야 하는데 이때 적용하는 기법이 non-max suppressio**

![Untitled](SPPNET%20Pri%205e4b0/Untitled%203.png)

- Non-max suppresion이 적용된 후 각각의 객체에 대한 bounding box가 1개만 남은 것을 확인할 수 있음

### ****Non-max suppression 알고리즘 작동 단계****

1. 하나의 클래스에 대한 bounding boxes 목록에서 가장 높은 점수를 갖고 있는 bounding box를 선택하고 목록에서 제거합니다. 그리고 final box에 추가합니다.

1. 선택된 bounding box를 bounding boxes 목록에 있는 모든 bounding box와 IoU를 계산하여 비교합니다. IoU가 threshold보다 높으면 bounding boxes 목록에서 제거합니다.

1. bounding boxes 목록에 남아있는 bounding box에서 가장 높은 점수를 갖고 있는 것을 선택하고 목록에서 제거합니다. 그리고 final box에 추가합니다.

1. 다시 선택된 bounding box를 목록에 있는 box들과 IoU를 비교합니다. threshold보다 높으면 목록에서 제거합니다.

1. bounding boxes에 아무것도 남아 있지 않을 때 까지 반복합니다.

1. 각각의 클래스에 대해 위 과정을 반복합니다.

ex)

[YOLO](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_4740)

# mAP(mean Average Precision)

- mAP(mean Average Precision)는 Faster R-CNN, SSD와 같은 object detector의 정확도를 측정하는 유명한 평가지표

→ mAP를 알아보기 전에 precision(정밀도), recall(재현율), IoU(intersection of union)에 대한 개념을 알아야함

### ****Precision(정밀도)와 recall(재현율)****

- Precision은 모델이 True라고 예측한 것 중 정답도 True인 것의 비율을 의미
- recall은 ****실제 정답이 True인 것중에서 모델이 True라고 예측한 것의 비율

- 공식

![Untitled](SPPNET%20Pri%205e4b0/Untitled%204.png)

- TP,TN,FP,FN

![Untitled](SPPNET%20Pri%205e4b0/Untitled%205.png)

ex)

- 이미지에 10개의 사과가 있고 모델이 6개의 사과를 검출하여 5개는 TP, 1개는 FP라고 가정
- 모델이 검출하지 못한 4개의 사과는 FN

⇒ Precision = 5/6가 되고, Recall = 5/10이 됨

→ Precision을 봤을 때는 좋은 성능을 갖고 있는 것 같은데 Recall을 보면 안좋은 성능을 갖고 있음

- 어느 한 값으로 알고리즘의 성능을 판단하기에는 불가능하고, **두 값을 종합해서 알고리즘을 평가하기 위한 것이 AP(Average Precision)**

### ****AP(Average Precision)****

- **AP는 Precision-Recall 그래프의 아래 면적**
- Precision-Recall 그래프?

ex) 

- 5개의 사과가 포함되어 있는 이미지에 Detector가 10개의 사과를 검출했다고 가정

![Untitled](SPPNET%20Pri%205e4b0/Untitled%206.png)

- recall은 서서히 증가하다가 rank 10에서 5번째 True때 1.0이 됨
- Precision은 들쑥날쑥 한 것을 확인할 수 있으며 5개의 사과중 10개의 사과를 검출했으므로 최종적으로 0.5의 precision가 됨

⇒ Recall에 따른 Precision을 Plot으로 그리면 Precision-Recall 그래프

![Untitled](SPPNET%20Pri%205e4b0/Untitled%207.png)

![Untitled](SPPNET%20Pri%205e4b0/Untitled%208.png)

- 곡선의 아래 면적이 AP(Average Precision)
- But, 보통 계산 전에 Precision-Recall 그래프를 바꿔줌

- 바꾼 다음 아래 면적을 구해서 AP를 구함

![Untitled](SPPNET%20Pri%205e4b0/Untitled%209.png)

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2010.png)

- AP = A1 + A2 + A3 + A4

with:

A1 = (0.0666-0) x 1 = 0.0666

A2 = (0.1333-0.0666) x 0.6666=0.04446222

A3 = (0.4 - 0.1333) x 0.4285 = 0.11428095

A4 = (0.4666 - 0.4) x 0.3043 = 0.02026638

AP = 0.0666 + 0.04446222 + 0.11428095 + 0.02026638

AP = 0.24560955

AP = 24.56%

# R-CNN

- R-CNN 중요한 아이디어
1. **region proposals로 object 위치를 알아내고, 이를 CNN에 입력하여 class를 분류**
2. **Larger data set으로 학습된 pre-trained CNN을 fine-tunning 함**

## R-CNN 등장 배경

- 2012년 image classification challenge에서 AlexNet이 큰 성공을 보이자 object detection에서도 CNN을 활용한 연구가 진행됩니다. 바로 그 결과물이 R-CNN이며, object detection 분야에 적용하기 위해 **region proposals와 CNN을 결합**한 모델
- R-CNN 이전에 sliding window detection에 CNN을 적용한 **OverFeat model**
이 존재
- OverFeat model은 bounding box가 정확하지 않고, 모든 sliding windows에 CNN을 적용하기 때문에 연산량이 많다는 문제점을 갖고 있어 현재 사용하지 않음

→ OverFeat model 설명

[[Paper Review] Introduction to Object Detection Task : Overfeat, RCNN, SPPNet, FastRCNN](https://www.youtube.com/watch?v=SMEtbrqJ2YI)

## R-CNN 알고리즘 단계

- 논문에 기재된 R-CNN의 전반적인 흐름

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2011.png)

- 아래는 R-CNN의 흐름을 더 보기 좋게 그린 도표

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2012.png)

- R-CNN 도식화

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2013.png)

1. 입력 이미지에 Selective Search 알고리즘을 적용하여 물체가 있을만한 bounding box(region proposal) 2000개를 추출
2. 추출된 bounding box를 227 x 227 크기로warp(resize)하여 CNN에 입력
3. fine tunning 되어 있는 pre-trained CNN을 사용하여 bounding box의 4096차원의 특징 벡터를 추출
4. 추출된 벡터를 가지고 각각의 클래스(Object의 종류) 마다 학습시켜놓은 SVM Classifier를 통과
5. bounding box regression을 적용하여 bounding box의 위치를 조정

### Object Detection Data

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2014.png)

### Selective Search

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2015.png)

- Selective Search란?
    - 전부를 탐색하는 Complete Search과는 달리 특정 기준에 따라 탐색을 실시함,
    - Bottom-Up의 탐색방법 중 하나인 계층적 그룹 알고리즘 등이 사용

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2016.png)

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2017.png)

- Selective Search의 절차
    - Selective Search은 초기의 작은 크기의 세분화 영역을 설정하고, 이를 계층적 그룹 알고리즘을 사용하여 병합하고, 이를 바탕으로 영역을 제안하는 단계론 진행

1. 이미지의 초기 세그먼트를 정하여, 수많은 region 영역을 생성
2. greedy 알고리즘을 이용하여 각 region을 기준으로 주변의 유사한 영역을 결합
3. 결합되어 커진 region을 최종 region proposal로 제안

### **Region Proposal**

- selective search 기법을 사용해서 이미지에서 object의 위치를 추출
- 이미지에 selective search를 적용하면 2000개의 region proposal이 생성
- CNN의 입력 사이즈(227x227)로 warp(resize) 하여 CNN에 입력
- 논문에서는 warp 과정에서 object 주변 16 픽셀도 포함하여 성능을 높였다고 함

### Feature Extraction(CNN)

- Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여 4096 크기의 특징 벡터를 추출

- ILSVRC 2012 데이터 셋으로 미리 학습된 pre-trained CNN 모델을 사용했으며, Object detection을 적용할 dataset으로 fine-tunning함

- fine tune 시에는 실제 Object Detection을 적용할 데이터 셋에서 ground truth에 해당하는 이미지들을 가져와 학습, 그리고 Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞춤

- CNN은 5개의 convulutional layer와 2개의 FC layer를 가진 AlexNet 형태를 사용했습니다. R-CNN은 soft-max layer 대신에 SVM을 사용하기 때문에 2개의 FC layer가 존재

→ fine tune을 적용했을 떄와 하지 않았을 때의 성능을 비교해보면 아래와 같습니다.

![https://blog.kakaocdn.net/dn/bklvjP/btqAQl2Z3K3/altDKimUjrdIaMiXocinv1/img.png](https://blog.kakaocdn.net/dn/bklvjP/btqAQl2Z3K3/altDKimUjrdIaMiXocinv1/img.png)

- 아래는 fine-tunning을 적용한 후와 적용하지 않았을 때의 성능입니다.
- 1~3 행은 fine-tunning을 적용하지 않았고, 4~6행은 fine-tunning을 적용
- FT는 fine tune의 약자이며, 각 CNN 레이어 층에서 추출된 벡터로 SVM Classifier를 학습시켜서 얻은 mAP를 비교
- 전반적으로 fine tuning을 거친 것들이 성능이 더 좋음을 확인

⇒ **정리하자면, 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출합니다.**

### **Classification(SVM)**

- CNN을 통해 추출한 벡터를 가지고 각각의 클래스 별로 SVM Classifier를 학습
- 주어진 벡터를 놓고 이것이 해당 물체가 맞는지 아닌지를 구분하는 Classifier 모델을 학습

- 이미 학습되어 있는 CNN Classifier를 두고 왜 SVM을 별도로 학습하는가?

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2018.png)

### **Bounding Box Regression**

- Selective Search를 통해서 찾은 박스 위치는 상당히 부정확

→ 성능을 끌어올리기 위해서 이 박스 위치를 교정해주는 부분을 Bounding Box Regression이라 함

- 하나의 박스를 다음과 같이 표기할 수 있음

(여기서 x, y는 이미지의 중심점, w, h는 각각 너비와 높이)

![https://blog.kakaocdn.net/dn/bdrbST/btqARhTcjKB/IERkqDTf3Ecq7ydiVpNk4k/img.png](https://blog.kakaocdn.net/dn/bdrbST/btqARhTcjKB/IERkqDTf3Ecq7ydiVpNk4k/img.png)

- Ground Truth에 해당하는 박스도 다음과 같이 표기할 수 있음.

![https://blog.kakaocdn.net/dn/zVIPj/btqAQ3On38L/I5Ml4DQNA2XeZwGDg3TtvK/img.png](https://blog.kakaocdn.net/dn/zVIPj/btqAQ3On38L/I5Ml4DQNA2XeZwGDg3TtvK/img.png)

- 목표는 P에 해당하는 박스를 최대한 G에 가깝도록 이동시키는 함수를 학습시키는 것
- 박스가 인풋으로 들어왔을 때, x, y, w, h를 각각 이동 시켜주는 함수들을 표현해보면 다음과 같음

![https://blog.kakaocdn.net/dn/bALrfw/btqAN4nGfTA/5kDiawSejYETki1m1TQOH0/img.png](https://blog.kakaocdn.net/dn/bALrfw/btqAN4nGfTA/5kDiawSejYETki1m1TQOH0/img.png)

- 이 때, x, y는 점이기 때문에 이미지의 크기에 상관없이 위치만 이동시켜주면 됨
- 반면에 너비와 높이는 이미지의 크기에 비례하여 조정을 시켜주어야 함

→ 이러한 특성을 반영하여 P를 이동시키는 함수의 식을 짜보면 다음과 같음

![https://blog.kakaocdn.net/dn/brT1sV/btqAOIj9OyD/CuGlIyoYusa64jUv0YpnoK/img.png](https://blog.kakaocdn.net/dn/brT1sV/btqAOIj9OyD/CuGlIyoYusa64jUv0YpnoK/img.png)

→ 학습을 통해서 얻고자 하는 함수는 d 함수

- 저자들은 이 d 함수를 구하기 위해서 앞서 CNN을 통과할 때 pool5 레이어에서 얻어낸 특징 벡터를 사용
- 그리고 함수에 학습 가능한 웨이트 벡터를 주어 계산

→ 이를 식으로 나타내면 아래와 같음

![https://blog.kakaocdn.net/dn/c6IqQU/btqAN3vwrBF/1fRsKyN9bJWZ3laKnloUo1/img.png](https://blog.kakaocdn.net/dn/c6IqQU/btqAN3vwrBF/1fRsKyN9bJWZ3laKnloUo1/img.png)

- 이제 웨이트를 학습시킬 로스 펑션을 세워보면 다음과 같음
- 일반적인 MSE 에러 함수에 L2 normalization을 추가한 형태
- 저자들은 람다를 1000으로 설정함

![https://blog.kakaocdn.net/dn/DchNe/btqAPeQHoXm/sOodgDOLkFOa08kdFoX5BK/img.png](https://blog.kakaocdn.net/dn/DchNe/btqAPeQHoXm/sOodgDOLkFOa08kdFoX5BK/img.png)

- 여기서 t는 P를 G로 이동시키기 위해서 필요한 이동량을 의미하며 식으로 나타내면 아래와 같음

![https://blog.kakaocdn.net/dn/dkWx6v/btqARw3Fnar/M8Gx32rXA13J2r58PLmc0K/img.png](https://blog.kakaocdn.net/dn/dkWx6v/btqARw3Fnar/M8Gx32rXA13J2r58PLmc0K/img.png)

⇒ **정리를 해보면 CNN을 통과하여 추출된 벡터와 x, y, w, h를 조정하는 함수의 웨이트를 곱해서**

**바운딩 박스를 조정해주는 선형 회귀를 학습시키는 것입니다.**

### **R-CNN에서 학습이 일어나는 부분**

1. 이미지 넷으로 이미 학습된 모델을 가져와 fine tuning 하는 부분
2. SVM Classifier를 학습시키는 부분
3. Bounding Box Regression

### 속도 및 정확도

- 테스트 시에 R-CNN은 이미지 하나 당 GPU에서는 13초, CPU에서 54초
- 속도 저하의 가장 큰 병목 구간은 selective search를 통해서 찾은 2천개의 영역에 모두 CNN inference를 진행하기 때문
- 정확도의 경우 Pascal VOC 2010을 기준으로 53.7%를 기록하

→ 당시의 기록들을 모두 갈아치우며 획기적으로 Object Detection 분야에 발전을 이끌었던 스코어

### ****문제점****

- R-CNN은 비효율성을 지니고 있음
    - 하나의 이미지에 2000개의 region이 존재할 때, R-CNN은 각각의 region마다 이미지를 cropping 한 뒤 CNN 연산을 수행하여 2000번의 CNN 연산을 진행하게 됨
    - 따라서 연산량이 많아지고 detection 속도가 느리다는 단점

# Bag of Visual Word

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2019.png)

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2020.png)

![Untitled](SPPNET%20Pri%205e4b0/Untitled%2021.png)

### 참고자료

- 유튜브

[[Paper Review] Introduction to Object Detection Task : Overfeat, RCNN, SPPNet, FastRCNN](https://www.youtube.com/watch?v=SMEtbrqJ2YI)

[십분딥러닝_10_R-CNN(Regions with CNNs)_1](https://www.youtube.com/watch?v=W0U2mf9pf8o)

[십분딥러닝_10_R-CNN(Regions with CNNs)_2](https://www.youtube.com/watch?v=FSZLcqEgq9Q)

- 블로그

[[Object Detection] IoU(Intersection over Union)를 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/402?category=968059)

[[Object Detection] 비-최대 억제(NMS, Non-maximum Suppression)를 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/403?category=968059)

[[Object Detection] mAP(mean Average Precision)을 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/407?category=968059)

[갈아먹는 Object Detection [1] R-CNN](https://yeomko.tistory.com/13)

[[논문 리뷰] R-CNN (2013) 리뷰](https://deep-learning-study.tistory.com/410?category=968059)