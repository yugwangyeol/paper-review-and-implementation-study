# SPPNET

https://www.notion.so/SPPNET-1717396cd0564e24b4824f24e2c2baba

# Motivation

1) Top  CNN operation

- RCNN은 Selective Search를 통해 대략 2000개의 candidnate bounding box를 만듬
- 2000개의 candidnate bounding box를 CNN을 입력하게되면 하나의 이미지를 학습하거 detection을 하는데 많은 시간이 소요됨

→  실시간이 불가능함

![Untitled](SPPNET%20de227/Untitled.png)

2) Distortion by warping

- RCNN은 candidnate bounding box내의 Feature를 뽑아내기 위해 AlexNEt을 사용
- candidnate bounding box가 227 x 227 size로 wapping이 됨
- Wapping과정에서 이미지 왜곡 현상이 발생

![Untitled](SPPNET%20de227/Untitled%201.png)

- crop을 적용하면 crop된 구역만 CNN을 통과시키기 때문에, 전체 이미지 정보가 손실이 발생

→ 위 그림만 보더라도 자동차 사진을 입력 크기로 맞춰주기 위해 자동차가 잘려나간 것을 확인할 수 있음

- warp을 적용하면 이미지에 변형이 일어남

→ 등대가 기존의 가로세로비를 유지하지 못하고 옆으로 퍼진 채로 CNN을 통과

- 그러면 왜 CNN크기는 고정되어야 하는가?
    - Convolution 필터들은 사실 입력 이미지가 고정될 필요가 없음
    - sliding window 방식으로 작동하기 때문에, 입력 이미지의 크기나 비율에 관계 없이 작동함
    - 입력 이미지 크기의 고정이 필요한 이유는 바로 컨볼루션 레이어들 다음에 이어지는 fully connected layer가 고정된 크기의 입력을 받기 때문임

⇒ 이러한 문제를 해결 하고자 SPPNet 등장

## SPPNet 핵심 아이디어

![Untitled](SPPNET%20de227/Untitled%202.png)

- SPPnet은 FC layer 이전에 Spatial Pyramid Pooling layer를 추가하여 convolutional layer가 임의의 사이즈로 입력을 취할 수 있게함
- 입력 사이즈 제한을 제거함으로써 입력 이미지를 crop / warp 필요성을 제거
- spatial pyramid pooling layer를 추가하여 crop/warp 단계를 제거

![Untitled](SPPNET%20de227/Untitled%203.png)

# Architecture

## 1) Reduce CNN operation

- R-CNN에서는 입력 이미지에서부터 Region proposal방식을 이용해  candidnate bounding box를 선별하고, 모든  candidnate bounding box에 대해서 CNN작업을 진행
- if 2000개의  candidnate bounding box가 나오게 되면 2000번의 CNN과정을 진행하게 됨

**BUT**

- SPP-Net은 입력 이미지를 먼저 CNN작업을 진행하고 다섯번쨰 Conv layer에 도착한 feature map을 기반으로 region proposal 방식을 적용해  candidnate bounding box를 선별
- 이렇게 되면 CNN 과정을 1번 거침
- RCNN 2000 → SPPNet 1번의 CNN operation 절감 효과가 나타남

![Untitled](SPPNET%20de227/Untitled%204.png)

## 2) Remove warping process and avoid distortion

- SPPNet은 warping으로 인한 disortion을 없애주고자 “spatial pyramid pooling”을 사용
- RCNN과의 차이는 warping하는 부분이 삭제되고 spatial pyramid pooling부분이 추가

![Untitled](SPPNET%20de227/Untitled%205.png)

## SPPNet 구조&알고리즘

![Untitled](SPPNET%20de227/Untitled%206.png)

![Untitled](SPPNET%20de227/Untitled%207.png)

(1) Selective Search를 사용하여 약 2000개의 region proposals를 생성합니다.

(2) 이미지를 CNN에 통과시켜 feature map을 얻습니다.

(3) 각 region proposal로 경계가 제한된 feature map을 SPP layer에 전달합니다.

(4) SPP layer를 적용하여 얻은 고정된 벡터 크기(representation)를 FC layer에 전달합니다.

(5) SVM으로 카테고리를 분류합니다.

(6) Bounding box regression으로 bounding box 크기를 조정하고 non-maximum suppression을 사용하여 최종 bounding box를 선별합니다.

## SPP Layer

![Untitled](SPPNET%20de227/Untitled%208.png)

- SPPnet은 5개 conv layer와 3개 fc layer를 활용
- SPP layer 위치는 conv5 layer 이후에 위치

![Untitled](SPPNET%20de227/Untitled%209.png)

- 우선 spatial bins의 개수를 선정

ex) 

1. 50 bin = [6x6, 3x3, 2x2, 1x1], 30 bin = [4x4, 3x3, 2x2, 1x1] 을 생각

→ [6x6, 3x3, 2x2, 1x1]은 conv5의 feature map에 pooling을 적용하여 생성되는 출력 크기

1. 위 그림에서는 21 bin = [4x4, 2x2, 1x1] 인 경우,  21 bin의 경우에 3개의 pooling 으로 이루어져있음

1. 각각의 pooling을 conv5 layer에 적용하여 특징을 추출하고 4x4, 2x2, 1x1의 크기를 출력

→ 일자로 피면 bin의 수

- 입력 사이즈가 다양하므로 conv5에서 출력하는 feature map의 크기도 다양하게 됨
- 다양한 feature에서 pooling의 window size와 stride 만을 조절하여 출력 크기를 결정
- window size = ceiling(feature map size / pooling size). stride = floor(feature map size / pooling size) 로 계산하면 어떠한 feature map 크기가 오더라도 고정된 pyramid size를 얻을 수 있음

![Untitled](SPPNET%20de227/Untitled%2010.png)

- [pool3x3] 의 window size = ceiling(13 / 3) = 5, stride = floor(13 / 3) = 4 로 설정
- 13x13 feature map에서 각각의 pooling의 window size, stride를 계산한 표

- SPP layer의 출력 차원은 k * M 이 됨
- k는 conv5 layer에서 출력한 feature map의 filter 수
- M은 사전에 설정한 bin의 수

- Figure3에서는 256개의 feature map, 21개 bin이므로 SPP layer는 256 * 21 차원 벡터를 출력
- 256 * 21 차원 벡터가 fc layer의 입력으로 통과
- feature map 크기와 관계 없이 bin과 feature map의 filter 수로 출력 차원을 계산하므로 고정된 차원 벡터를 갖게됨
- 다양한 입력 이미지 크기를 입력 받아 다양한 feature map size가 생성되고 SPP layer를 거쳐서 고정된 크기의 벡터가 생성

![Untitled](SPPNET%20de227/Untitled%2011.png)

# Result

![Untitled](SPPNET%20de227/Untitled%2012.png)

- SPPNet BB모델이 가장 좋은 성능을 보임

## SPPNet 장점

1. Spartial Pyramid Pooling을 통해 RCNN에서 사용된 warping 작업을 없애 distortion을 피할 수 있음
2. RCNN에서는 CNN연산을 2000번 한것에 비해, SPPNet은 CNN연산을 한 번만 하면서, Training,Test시간을 굉장히 단축 시킴

## SPPNet 단점

1. end-to-end 방식이 아니라 학습에 여러 단계가 필요함.(fine-tuning, SVM training, Bounding Box Regression)
2. 여전히 최종 클래시피케이션은 binary SVM, Region Proposal은 Selective Search를 이용
3. fine tuning 시에 SPP를 거치기 이전의 Conv 레이어들을 학습 시키지 못한다. 단지 그 뒤에 Fully Connnected Layer만 학습

→ 세 번째 한계점에 대해서 저자들은 ***"for simplicity"***  라고만 설명

# 참고자료

- 블로그

[6. SPP Net](https://89douner.tistory.com/89)

[[논문 리뷰] SPPnet (2014) 리뷰, Spatial Pyramid Pooling Network](https://deep-learning-study.tistory.com/445)

[갈아먹는 Object Detection [2] Spatial Pyramid Pooling Network](https://yeomko.tistory.com/14)

[[논문정리] SPPNet : Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://n1094.tistory.com/30)

- 유튜브

[[Paper Review] Introduction to Object Detection Task : Overfeat, RCNN, SPPNet, FastRCNN](https://www.youtube.com/watch?v=SMEtbrqJ2YI)

[천우진 - Spatial pyramid pooling in deep convolutional networks for visual recognition](https://www.youtube.com/watch?v=i0lkmULXwe0)

- 깃허브

[https://github.com/mmmmmmiracle/SPPNet](https://github.com/mmmmmmiracle/SPPNet)

[keras-transfer-learning-for-oxford102/bootstrap.py at master · Arsey/keras-transfer-learning-for-oxford102](https://github.com/Arsey/keras-transfer-learning-for-oxford102/blob/master/bootstrap.py)

**→ 코드 결과**

![Untitled](SPPNET%20de227/Untitled%2013.png)
