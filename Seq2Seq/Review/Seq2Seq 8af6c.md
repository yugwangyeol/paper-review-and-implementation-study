# Seq2Seq

# Sequence to Sequence Learning with Nural Network(NIPS 2014)

- 본 논문에서는 LSTM을 활용한 Seq2Seq 기계번역을 제안함
    - Seq2Seq는 딥러닝 기반 기계번역의 돌파구와 같은 역할을 수행
    - Transformr가 나오기 전까지 state-of-the art로 사용 됨

![Untitled](Seq2Seq%208af6c/Untitled.png)

# 딥러닝 기반의 기계번역 발전 과정

- 2021년 기준으로 최신 고성능 모델들을 Transformer 아키텍처를 기반으로 함
    - GPT: Transformer의 Decoder 아키텍처를 활용
    - BERT: Transformer의 인코더 아키텍처를 활용

![Untitled](Seq2Seq%208af6c/Untitled%201.png)

# 언어 모델(Language Model)

- 언어 모델이란 문장(시퀀스)에 확률을 부여하는 모델을 의미함
- 언저 모델을 가지고 특정한 상황에서의 적절한 문장이나 단어를 예측할 수 있음
    - 기계 번역 예시
        - P(난 널 사랑해 | I love you) > P(난 널 싫어해 | I love you)
    - 다음 단어 예측 예시
        - P(먹었다 | 나는 밥을) > P(싸웠다 | 나는 밥을)
    
- 하나의 문장(W)은 여러 개의 단어(w)로 구성됨
    - $P(W) = P(w_1, w_2,w_3, ... , w_n)$
    - P(친구와 친하게 지낸다) = P(친구와,친하게, 지낸다)
    
- 연쇄법칙 (Chain Rule)
    
    
    ![Untitled](Seq2Seq%208af6c/Untitled%202.png)
    
    - p(친구와 친하게 지낸다) = P(친구와) * P(친하게 | 친구와) * P(지낸다 | 친구와 친하게)
    
- 전통적인 통계적 언어 모델은 카운트 기반으로 접근을 사용

![Untitled](Seq2Seq%208af6c/Untitled%203.png)

- 현실에서 모든 문장에 대한 확률을 가지고 있으려면 매우 방대한 양의 데이터가 필요
- 긴 문장을 처리하기 어려움
    
    
    ![Untitled](Seq2Seq%208af6c/Untitled%204.png)
    

- 현실적인 해결책으로 N-gram 언어 모델이 사용됨
    - 인접한 일부 단어만 고려함

# 전통적인 RNN 기반의 번역 과정

- 전통적인 RNN 기반의 기계 번역은 입력과 출력의 크기가 같다고 가정
    - 입력:  $(x_1, ..., x_T)$
    - 출력: $(y_1, ... , y_T)$
        - $h_t = sigmoid(W^{hx}x_t + W^{hh}h_{t-1})$
        - $y_t = W^{yh}h_t$

![Untitled](Seq2Seq%208af6c/Untitled%205.png)

# RNN 기반의 Sequence to Sequence 개요

- 전통적인 초창기 RNN 기반의 언어 모델은 다양한 한계점을 가지고 있음
    - 이를 해결하기 위해 인코더가 고정된 크기의 문맥 벡터(Context vector)를 추출
    - 이후에 문맥 벡터로부터 디코더가 번역 결과를 추론
    - 본 Seq2Seq 논문에서는 LSTM를 이용해 문맥 벡터를 추출하도록 하여 성능을 향상 시킴
        - 인코더 마지막 hidden state만을 Contect Vector로 사용

![Untitled](Seq2Seq%208af6c/Untitled%206.png)

- 인코더와 디코더는 서로 다른 파라미터(가중치)를 가짐

# RNN 기반의 Seqence to Sequence  자세히 살펴보기

![Untitled](Seq2Seq%208af6c/Untitled%207.png)

- RNN 기반 Seq2Seq 모델의 목표 공식(Formulation)은 다음과 같음
    
    
    ![Untitled](Seq2Seq%208af6c/Untitled%208.png)
    

# Seq2Seq의 성능 개선 포인트: LSTM 활용 및 입력 문장의 순서 뒤집기

- 기본적인 RNN대신에 LSTM을 활용했을 때 더 높은 정확도를 보임
- 실제 학습 및 테스트 과정에서 입력 문장의 순서를 거꾸로 했을 때 더 높은 정확도를 보임
    - 출력 문장의 순서는 바꾸지 않음

![Untitled](Seq2Seq%208af6c/Untitled%209.png)

# Papepr Seq2Seq

## Introduction

- DNN (Deep Neural Network)는 음성 인식, 사물 인식 등에서 꾸준한 성과를 내어왔음
- 하지만 input size가 fixed된다는 한계점이 존재하기 때문에 sequencial problem을 제대로 해결할 수 없다는 한계점이 존재

- 논문에서는 2개의 LSTM (Long Short Term Memory)을 각각 Encoder, Decoder로 사용해 sequencial problem을 해결하고함
- 이를 통해 많은 성능 향상을 이루어냈으며, 특히나 long sentence에서 더 큰 상승 폭을 보임
- 단어를 역순으로 배치하는 방식으로도 성능을 향상시켰다.

## The Model

- RNN은 기본적으로 sequencial problem에 매우 적절한 model임
- input size와 output size가 다른 경우에 대해서는 좋은 성능을 보일 수 없으며, 또한 **장기 의존성 문제가 발생할 수 있음**
- **LSTM은 장기적 의존성 문제 또한 학습할 수 있다**고 알려져있음

→  LSTM은 이러한 전략을 성공적으로 수행할 수 있을 것이다.

![Untitled](Seq2Seq%208af6c/Untitled%2010.png)

- LSTM은 "A", "B", "C", "<EOS>"의 표현을 계산한 다음 이 표현을 사용하여 "W", "X", "Y", "Z", "<EOS>"의 확률을 계산

![Untitled](Seq2Seq%208af6c/Untitled%2011.png)

- 본 논문에서 제시하는 model은 Encoder LSTM에서 하나의 context vector를 생성한 뒤 Decoder LSTM에서 context vector를 이용해 output sentence를 생성하는 방식으로 RNN의 한계점을 극복하고자 했음
- **input과 output sentence 간의 mapping을 하는 것이 아닌, input sentence를 통해 encoder에서 context vector를 생성하고, 이를 활용해 decoder에서 output sentence를 만들어냄**

- Encoder LSTM의 output인 context vector는 Encoder의 마지막 layer에서 나온 output임
- 이를 Decoder LSTM의 첫번째 layer의 input으로 넣게 되며, 주목할만한 점은 input sentence에서의 word order를 reverse해 사용했다는 것
- 또한 (End of Sentence) token을 각 sentence의 끝에 추가해 variable length sentence를 다뤘음

## Seq2Seq모델의 세 가지 중요한 면

1. 서로 다른 두 가지 LSTM을 사용
    - 하나는 **input sequence용이고 다른 하나는 output sequence용**
    - 그렇게 하는 것이 학습해야 할 **model parameter의 수는 거의 증가시키지 않으면서도 LSTM이 다양한 언어쌍을 동시에 학습하는 것이 가능해지기 때문**
    
    ex)  English 언어 타입의 Input Sequence를 Representation Vector로 바꾸는 Encoder(LSTM 모델)에 해당 Representation Vector를 특정 언어 타입(French, Korean)의 Output Sequence로 바꾸는 Decoder들을 쌍으로 묶을 수 있다는 말
    
    ⇒ English->French, English->Korean
    

1. **깊은 LSTM이** **얕은 LSTM보다** **성능이** **훨씬** **뛰어나서 4개의** **레이어가** **있는 LSTM을** **선택**

1. **입력 문장의 단어 순서를 반대로**

ex) 문장 **a, b, c**를 문장 α, β, γ에 매핑하는 대신 LSTM에 **c, b, a**를 α, β, γ로 매핑하도록 요청합니다. 여기서 α, β, γ는 a, b, c의 번역

⇒ a는 α에 아주 가깝고, b는 β에 아주 가까워지므로(Encoder에서는 역순 input sequence를 넣기 때문에 **a가 마지막으로 들어가게 되며** Decoder에서는 **α가 가장 먼저 나오니까** a와 α를 가장 가까이 위치하도록 만든셈

⇒SGD가 input과 output 사이의 연관 관계를 연결하여 계산하는 것을 쉽게 만들어줌

![Untitled](Seq2Seq%208af6c/Untitled%2012.png)

## Conclusion

1. 우리는 어휘가 제한적이고 문제 구조에 대한 가정을 거의 하지 않는 대규모 **deep LSTM이 대규모 기계번역 작업에서 어휘가 무제한인 표준 SMT 기반 시스템보다 성능이 우수함**을 보여줌

1. **source sentences의 단어를 역순으로 배치 → 단기 종속성이 학습 문제를 더 쉽게 만들어줌**

⇒ 때문에 가장 많은 단기 종속성을 갖는 encoding 문제를 찾는 것이 중요하다고 결론을 내림

1. **매우 긴 문장을 정확하게 번역하는 LSTM의 능력**
    
    LSTM이 제한된 메모리로 인해 긴 문장에서 실패할 것이라고 확신했지만 **역 데이터셋으로 훈련된 LSTM**은 긴 문장을 번역하는 데 어려움이 거의 없었습니다
    
2. 마지막으로 단순하고 간단하며 상대적으로 최적화되지 않은 접근 방식이 SMT 시스템보다 성능이 우수하므로 추가 작업으로 번역 정확도가 훨씬 더 높아질 수 있다는 점

# 참고 자료

- 유튜브

[[딥러닝 기계 번역] Seq2Seq: Sequence to Sequence Learning with Neural Networks (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=4DzKM0vgG1Y)

- 블로그

[[논문 읽기] PyTorch 구현 코드로 살펴보는 Seq2Seq(2014), Sequence to Sequence Learning with Neural Networks](https://deep-learning-study.tistory.com/685?category=1004754)

[[NLP | 논문리뷰] Sequence to Sequence Learning with Neural Networks](https://velog.io/@xuio/NLP-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Sequence-to-Sequence-Learning-with-Neural-Networks)

[[논문 리뷰] Sequence to Sequence Learning with Neural Networks (2014 NIPS)](https://misconstructed.tistory.com/47)

[[논문리뷰] Sequence to sequence learning with neural networks](https://coshin.tistory.com/47)

[[NLP 논문 리뷰] Sequence To Sequence Learning With Neural Networks (Seq2Seq)](https://cpm0722.github.io/paper-review/sequence-to-sequence-learning-with-neural-networks)

[https://roasted-rake-be8.notion.site/Seq2Seq-2a6c05bd7c3b44ec863ab730bdf72d49](https://www.notion.so/Seq2Seq-2a6c05bd7c3b44ec863ab730bdf72d49)

- 깃허브

[Deep-Learning-Paper-Review-and-Practice/Sequence_to_Sequence_with_LSTM_Tutorial.ipynb at master · ndb796/Deep-Learning-Paper-Review-and-Practice](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_LSTM_Tutorial.ipynb)