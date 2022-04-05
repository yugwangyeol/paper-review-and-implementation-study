# Transformer Prior Learning(Attention)

# Attention Mechanism

1. Attention Mechanism의 정의
- 인간의 시각적 집중(Visual Attention)현상을 구현하기 위한 신경망적 기법을 말함

![Untitled](Transforme%2067f34/Untitled.png)

1. 가중치와 Attention의 공통점과 차이점
- 가중치와 Attention 모두 해당 값을 얼마나 가중시킬 것인가를 나타는 역할을 함
- Attention은 가중치와 달리 전체 또는 영역의 입력값을 반영하여, 그 중에 어떤 부분에 집중해야 하는지를 나타낸는것을 목표로 함

# Motivation

![Untitled](Transforme%2067f34/Untitled%201.png)

- 어텐션 매커니즘은 기계번역(machine translation)을 위한 sequence-to-sequence 모델(S2S)에 처음 도입
- 소스랭귀지(A,B,C)를 입력으로 해서 벡터로 만드는 앞부분을 인코더(encoder), 인코더가 출력한 벡터를 입력으로 해서 타겟랭귀지(W,X,Y,Z)를 출력하는 뒷부분을 디코더(decoder)라고 함

![Untitled](Transforme%2067f34/Untitled%202.png)

→ 소스랭귀지와 타겟랭귀지의 길이가 길어질 수록 모델의 성능이 나빠짐

→ W를 예측할 때 A,B,C 모두에 집중해 보게 되면 정확도가 떨어질 수 있음

- Seq2Seq모델의 한계

→ 고정된 크기의 벡더에 모든 정보를 압축함으로써 발생하는 정보 손실

# Attention 핵심 아디디어

- 디코더에서 출력 단어를 예측하는 매 시점(Time Step)마다, 인코더에서의 전체 입력 문장을 다시 한번 참고
- 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서 봄
- 다양한 attention 종류가 있지만, Dot-Product Attention과 다른 attention들의 차이는 attention score함수라는 중간 수식의 차이
- 양방향 RNN Encoder와, 단방향 RNN decoer를 사용

# Attention 알고리즘

![Untitled](Transforme%2067f34/Untitled%203.png)

- 출력 단어를 예측하기 위해서 인코더의 모든 입력 단어들의 정보를 다시 한번 참고
- Encoder hidden state에서는 양방향 LSTM이 사용

→ 한번은 정방향, 한번은 반대방향

![Untitled](Transforme%2067f34/Untitled%204.png)

## Dot-Product Attention(Attention Score)

![Untitled](Transforme%2067f34/Untitled%205.png)

1. Attention Score

![Untitled](Transforme%2067f34/Untitled%206.png)

- 디코딩 할 때, 인코더의 어느 입력시점에 집중할 것인지를 점수화한것

- 한 디코딩 시점으로부터 모든 입력 시점에 대해서 계산이 되고, 이 계산은 모든 디코딩 스텝에서 반복
- $e_{ij}$는 디코더가 i번째 단어를 예측할 때 쓰는 직전 스텝의 히든스테이트 벡터 $s_{i-1}$이 인코더의 j번째 열벡터 $h_j$와 얼마나 유사한지를 나타내는 스코어 값

![Untitled](Transforme%2067f34/Untitled%207.png)

- 디코더의 hidden state 값(st)을 transpose하고 각 hidden state(hi)와 내적 (dot product)하여 스칼라 값으로

![Untitled](Transforme%2067f34/Untitled%208.png)

1. Attention distribution

![Untitled](Transforme%2067f34/Untitled%209.png)

- Attention score를 가지고 softmax 함수를 통과시켜 확률화(분수화)하여 값을 만듦

- 계산된 각 0~1 사이의 값들이 바로 입력 시점에 대한 가중치, 즉 “시간의 가중치”가 되는 것
- $e_{ij}$에 소프트맥스 함수를 적용해 합이 1이 되도록 확률값으로 변환
- $T_x$는 인코더 입력 단어의 수를 가리킴

![Untitled](Transforme%2067f34/Untitled%2010.png)

1. Attention output

![Untitled](Transforme%2067f34/Untitled%2011.png)

- 인코더의 hidden state와 attention 가중치 값들을 곱하고, 최종적으로 더함

- 가중치 합을 계산하여 최종적으로 하나의 벡터로

- 이 벡터는 매 디코딩 시점마다 다르며, 따라서 기존에 fixed-length vector의 문제점을 해결할 수 있도록 구성

![Untitled](Transforme%2067f34/Untitled%2012.png)

1. Decoder hidden state

![Untitled](Transforme%2067f34/Untitled%2013.png)

- 계산해낸 Attention outout 벡터와 이전 디코더의 hidden state, 출력을 이용하여 최종적으로 다음 decoder hidden state를 출력

- fixed-length 벡터가 매 시간 인덱스마다 다르게 반영되는 것을 확인

![Untitled](Transforme%2067f34/Untitled%2014.png)

## Attention Mechanism의의

1) 확률과 에너지를 기반으로한 접근

![Untitled](Transforme%2067f34/Untitled%2015.png)

![Untitled](Transforme%2067f34/Untitled%2016.png)

- $\alpha_{ij}$를 확률적인 가중치로, $e_{ij}$를 해당 확률을 형성하는 기반인 에너지로 볼 수 있음

2) 긴 거리에서 의존성(Long Dependencies) 문제를 해결

![Untitled](Transforme%2067f34/Untitled%2017.png)

# Conclusion

- Attention: 모델이 다음 target word를 생성하는 것과 관련 있는 정보에만 집중하게 함
- 고정된 크기의 벡터에 input삾의 정보를 담아야 했던 기존의 Seq2Seq와는 달리 문장의 길이에 robust함
- 모델의 각각의 target word와 관련 있는 단어 or 그 annotation을 올바르게 정렬하게 함

# 참고 자료

- 블로그

[어텐션 매커니즘](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/)

[[논문 스터디] Attention: Neural Machine Translation by jointly Learning to Align and Translate](https://hong-yp-ml-records.tistory.com/63)

- 유튜브

[십분딥러닝_12_어텐션(Attention Mechanism)](https://www.youtube.com/watch?v=6aouXD8WMVQ)