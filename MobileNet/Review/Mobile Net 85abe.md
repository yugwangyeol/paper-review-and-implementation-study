# Mobile Net

https://www.notion.so/Mobile-Net-95ac902eed494cd2bc0fc1efc858d32a

# Abstract

- **MobileNet은 Depthwise separable convolution을 활용하여 모델을 경량화함**
- Xception은 Depthwise separable convolution을 활용하여 감소한 파라미터 수 많큼 층을 쌓아 성능을 높이는데 집중 → MobileNet은 반대로 경량화에 집중

![Untitled](Mobile%20Net%2085abe/Untitled.png)

- MobileNet이 경량화에 집중한 이유는 핸드폰이나 임베디드 시스템 같이 저용량 메모리환경에 딥러닝을 적용하기 위해서는 모델 경량화가 필요하기 때문
- **메모리가 제한된 환경에서 MobileNet을 최적으로 맞추기 위해 두 파라미터 latency와 accuracy의 균형을 조절**

# ****1. Depthwise Separable Convolution****

- **Depthwise Separable Convolution은 Depthwise Convolution 이후에 Pointwise Convolution을 결합한 것**

![Untitled](Mobile%20Net%2085abe/Untitled%201.png)

### **(1) Depthwise convolution**

- **Depthwise convolution은 각 입력 채널에 대하여 3x3 conv 하나의 필터가 연산을 수행하여 하나의 피쳐맵을 생성**
- 입력 채널 수가 M개이면 M개의 피쳐맵을 생성함
- 각 채널마다 독립적으로 연산을 수행하여 spatial correlation을 계산하는 역할
    
    → 예를 들어, 5 채널의 입력값이 입력되었으면, 5개의 3x3 conv가 각 채널에 대하여 연산을 수행하고, 5개의 feature map을 생성
    

Depthwise convolution의 연산량은 다음과 같음

![Untitled](Mobile%20Net%2085abe/Untitled%202.png)

- Dk는 입력값 크기, M은 입력의 채널수, DF는 피쳐맵 크기

### **(2) Pointwise convolution**

- **Pointwise convolution은 Depthwise convolution이 생성한 피쳐맵들을 1x1conv로 채널 수를 조정**
- 1x1conv는 모든 채널에 대하여 연산하므로 cross-channel correlation을 계산하는 역할

Pointwise convolution의 연산량은 다음과 같습니다.

![Untitled](Mobile%20Net%2085abe/Untitled%203.png)

- M은 입력 채널 수, N은 출력 채널 수, DF는 피쳐맵 크기

### ****(3) Depthwise separable convolution****

- Depthwise separable convolution은 Depthwise convolution 이후에 Pointwise convolution을 적용
- 아래 그림은 **MobileNet에서 사용하는 Depthwise separable convolution 구조**

![Untitled](Mobile%20Net%2085abe/Untitled%204.png)

![Untitled](Mobile%20Net%2085abe/Untitled%205.png)

![Untitled](Mobile%20Net%2085abe/Untitled%206.png)

![Untitled](Mobile%20Net%2085abe/Untitled%207.png)

![Untitled](Mobile%20Net%2085abe/Untitled%208.png)

![Untitled](Mobile%20Net%2085abe/Untitled%209.png)

이 과정을 시각화하면 다음과 같음

![Untitled](Mobile%20Net%2085abe/Untitled%2010.png)

- 큐브는 3차원의 필터 모양(혹은 parameter의 개수)을 나타내며, 표준 conv는 딱 봐도 큐브의 부피 합이 커 보이지만 Depthwise convolution와 Pointwise convolution는 하나 또는 2개의 차원이 1이므로 그 부피가 작음(즉, parameter의 수가 많이 적음).

- 이는 다르게 말해서 3차원적인 계산을 두 방향의 차원으로 먼저 계산한 후 나머지 한 차원을 그 다음에 계산하는 방식이라 생각해도 됨

전체 연산량

![Untitled](Mobile%20Net%2085abe/Untitled%2011.png)

- 둘의 연산량을 더해준 것이 됨
- **Depthwise separable convolution 연산량은 기존 conv 연산량보다 8~9배 더 적음**

기존 Conv 연산량은 다음과 같음

![Untitled](Mobile%20Net%2085abe/Untitled%2012.png)

- M채널의 Dk x Dk 크기의 입력값에 M개의 3x3conv 묶음을 N번 수행하여 DF x DF 크기의 피쳐맵을 생성

→ MobileNet은 이 Depthwise separable convolution을 기반으로 구축된 모델

# ****2. MobileNet Architecture****

![Untitled](Mobile%20Net%2085abe/Untitled%2013.png)

![Untitled](Mobile%20Net%2085abe/Untitled%2014.png)

- MobileNet 모델에서 Pointwise convolution이 비교적 많은 연산량과 파라미터가 사용된다는 것을 볼 수 있음

- 첫 번째 conv를 제외하고 depthwise separabel convolution을 사용
- 마지막 FC 레이어를 제외하고 모든 레이어에 BN, ReLU를 사용합니다. Down-sampling은 depthwise convolution과 첫 번째 conv layer에서 수행합니다. 총 28 레이어를 갖음

# 3. Hyper - parameter

MobileNet은 모델의 latency와 accuracy를 조절하는 두 개의 하이퍼파라미터가 존재합니다.

### **(1) Width Multiplier: Thinner Models**

- **첫 번째 하이퍼파라미터 αα는 MobileNet의 두께를 결정합니다. conv net에서 두께는 각 레이어에서 필터수를 의미함**
- 이 width Multiplier $\alpha$는 더 얇은 모델이 필요할 때 사용
- 입력 채널 M과 출력 채널 N에 적용하여 $\alpha M , \alpha N$이 됨

연산량은 다음과 같음

![Untitled](Mobile%20Net%2085abe/Untitled%2015.png)

- $\alpha = 0 \sim 1$ 범위이고 기본 MobileNet은 1을 사용
- Width Multiplier를 낮추면 모델의 파라미터 수가 감소

![Untitled](Mobile%20Net%2085abe/Untitled%2016.png)

### **(2) Resolution Multiplier: Reduced Representation**

- 두 번째 하이퍼파라미터는 Resolution Multiplier ρρ
- **모델의 연산량을 감소시키기 위해 사용**
- **ρρ는 입력 이미지에 적용하여 해상도를 낮춤**
- 범위는 0~1이고, 논문에서는 입력 이미지 크기가 224, 192, 169, 128 일때 비교함
- 기본 MobileNet은 ρρ=1을 사용

![Untitled](Mobile%20Net%2085abe/Untitled%2017.png)

### ****4. Comparison With Popular Models****

- MobileNet을 여러 multiplier 등 여러 세팅을 바꿔가면서 실험한 결과인데, 주로 성능 하락은 크지 않으면서도 모델 크기나 계산량이 줄었음을 보여줌
- 혹은 정확도는 낮아도 크기가 많이 작기 때문에 여러 embedded 환경에서 쓸 만하다는 주장

![Untitled](Mobile%20Net%2085abe/Untitled%2018.png)

- Depthwise Separable과 Full Convolution의 차이는 명확하다. 정확도는 1% 낮지만, **모델 크기는 7배 이상** 작음
- 또 Narrow와 Shallow MobileNet을 비교하면 아래와 같음(깊고 얇은 모델 vs 얕고 두꺼운 모델)

![Untitled](Mobile%20Net%2085abe/Untitled%2019.png)

- 계산량과 성능 사이의 trade-off는 위처럼 나타난다. 계산량이 지수적으로 늘어나면, 정확도는 거의 선형적으로 늘어남

![Untitled](Mobile%20Net%2085abe/Untitled%2020.png)

- 정확도, 계산량, 모델 크기를 종합적으로 비교

![Untitled](Mobile%20Net%2085abe/Untitled%2021.png)

- 웹에서 얻은 대량이지만 noisy한 데이터를 사용하여 학습한 다음 Stanford Dogs dataset에서 테스트해봄

![Untitled](Mobile%20Net%2085abe/Untitled%2022.png)

- MobileNet의 또 다른 쓸모 있는 점은 전혀 또는 거의 알려져 있지 않은(unknown or esoteric) 학습 과정을 가진 큰 모델을 압축할 수 있다는 것
- MobileNet 구조를 사용하여 얼글 특징 분류기에서 distillation을 수행했는데, 이는 분류기가 GT label 대신에 더 큰 모델의 출력값을 모방하도록 학습하는 방식으로 작동
- 기본 모델에 비해 최대 99%까지 연산량을 줄이면서도 성능 하락은 별로 없는 것을 볼 수 있음

![Untitled](Mobile%20Net%2085abe/Untitled%2023.png)

- MobileNet을 물체 인식에도 적용시켜서 Faster-RCNN 등과 비교
- 이 결과 역시 모델 크기나 연산량에 비해 성능이 좋다는 것을 보여주고 있음

![Untitled](Mobile%20Net%2085abe/Untitled%2024.png)

- 얼굴인식 모델에서 FaceNet은 SOTA 모델인데, 적절히 distillation을 수행한 결과, 성능은 조금 낮으나 연산량을 고려하면 만족할 만한 수준인 것 같음

![Untitled](Mobile%20Net%2085abe/Untitled%2025.png)

# 4****. Conclusion****

- Depthwise Separable Convolutions을 사용한 경량화된 모델 MobileNet을 제안
- 모델 크기나 연산량에 비해 성능은 크게 떨어지지 않고, 시스템의 환경에 따라 적절한 크기의 모델을 선택할 수 있도록 하는 여러 옵션(multiplier)를 제공

# 참고 자료

- 유튜브

[[AI 논문 해설] 모바일넷 MobileNet 알아보기](https://www.youtube.com/watch?v=vi-_o22_NKA)

[PR-044: MobileNet](https://www.youtube.com/watch?v=7UoOFKcyIvM)

- 블로그

[[논문 읽기] MobileNet(2017) 리뷰, Efficient Convolutional Neural Networks for Mobile Vision Applications](https://deep-learning-study.tistory.com/532)

[https://roasted-rake-be8.notion.site/MobileNet-13d579a78cc04ea3b243bcfc0420efa6](https://www.notion.so/MobileNet-13d579a78cc04ea3b243bcfc0420efa6)

- 깃허브

[https://github.com/cyrilminaeff/MobileNet](https://github.com/cyrilminaeff/MobileNet)
