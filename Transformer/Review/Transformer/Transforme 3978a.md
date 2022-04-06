# Transformer

# Transformer

![Untitled](Transforme%203978a/Untitled.png)

- Transforemr는 RNN이나 CNN을 사용하지 않고 오로지 Attention만 사용함
    
    → RNN과 CNN을 사용하지 않아 순서 정보를 주기 어려움
    
    → 대신 Positional Encoding을 사용함
    
- BERT와 같은 향상된 네트워크에서도 채택

- Encoder와 Decoder로 구성됨
    - Attention과정을 여러 레이어에서 반복하여 수행

# Transformer 동작 원리

## 입력값 임베딩

![Untitled](Transforme%203978a/Untitled%201.png)

- 임베딩 과정을 거침
- 행: 단어의 갯수, 열: 임베딩 차원의 수와 같음
- 논문에서는 512로 사용

![Untitled](Transforme%203978a/Untitled%202.png)

- RNN을 사용하지 않으려면 위치 정보를 포함하고 있는 임베딩을 사용해야 함
    - 위치정보: 단어가 앞에 오는지 뒤에 오는지를 말함
    - 이를 위해서 Transformer는 Positional Encoding을 사용함
- Positional Encoding과 Input Embedding Matrix를 더하여 위치 정보를 포함함

## Encoding

![Untitled](Transforme%203978a/Untitled%203.png)

- 임베딩이 끝난 후에는 Attention을 진행함
- Encoder에서의 Attention은 self-Attention
    - 문장내의 각 단어끼리의 연관성을 가지고 있는지 구함
    - Attention score를 구함
- 문맥에 대한 정보를 잘 학습하도록 만듬

![Untitled](Transforme%203978a/Untitled%204.png)

- 성능 향상을 위해 잔여 학습(Residual Learning)을 사용

![Untitled](Transforme%203978a/Untitled%205.png)

- Attention 과정과 Normalization 과정을 반복함
- 각 레이어는 서로 다른 파라미터를 가짐
- 입력, 출력 차원이 같음

## Encoder와 Decoder

![Untitled](Transforme%203978a/Untitled%206.png)

- 여러개의 인코더를 반복, 마지막 인코더 출력값이 디코더로 들어감
- 입력 소스 문장 중에서 어떤 단어에 가장 많은 초점을 두어야 하는지 알려주기 위함

### 디코더

- 매번 인코더값을 디코더에 넣어줌
- Positional Encoding을 통해 단어들의 상대적인 위치를 넣어줌

- Attention이 두개 존재
- 첫번쨰 Attention은 self attetion으로 디코더 입력값의 서로가 서로에게 어떤 가중치를 가지는지 확인
- 두번째 Attention은 인코더의 정보를 Attention학 만듬
    - 각각의 출력되고 있는 정보가 소스 단어와 어떤 관계를 가지는지 확인
    - Encoder Decoder Attention이라 부름
    
    ex) Eecoer 압력값: 선생님 → Encoder : I, am, a, teacher
    
    → 선생님과 teacher가 가장 관계가 높음을 확인
    
- 입력 차원과 출력 차원이 같음

![Untitled](Transforme%203978a/Untitled%207.png)

- 마지막 Encoder layer의 출력이 모든 Decoder layer에 입력
    - 위 그림은 n_layer=4일떄
    - 대부분 Encoder와 Decoder층이 같음
    - 두 번째 Attention에 입력

![Untitled](Transforme%203978a/Untitled%208.png)

- Transformer도 Encoder아 Decoder를 사용
- RNN을 사용하지 않고 Encoder와 Decoder를 다수 사용하는게 특징
- <EOS>가 나올때까지 Decoder를 사용함

- 입력 단어 자체가 하나로 연결되어 한번에 입력, 한번에 Attention값을 구함
- 병렬적 처리

## Multi head Attention

![Untitled](Transforme%203978a/Untitled%209.png)

- 왼쪽 그림이 multi head attention
- 중간에 Scaled Dot-Product Attention이 존재

- Attention위한 세가지 입력 요소
    - Query
    
    → 물어보는 주체 (teacher)
    
    - key
    
    →물어보는 대상 (i,am,a teacher)
    
    - Value
    
    → 구해진 확률값과 실제 값을 곱함
    

- 입력값은 서로 다른 value, key, query로 h개로 구분됨
- 서로 다른 attetion컨셉을 잡기 위함
- h는 head의 갯수
- 입력값과 출력값이 같아야 하므로 concat을 진행

ex) encoder decoer attention (Decoder scond attention)

- Decoder 출력값 각각이 Query
- Encoder 출력값이 Key, Value

![Untitled](Transforme%203978a/Untitled%2010.png)

- query, key, value값을 이력 받음
- query와 key를 곱해서 에너지를 구하고, 확률값으로 표현, d(key 차원)로 스케일링
- softmax함수 때문에 스케일링 진행

→ 각각의 query가 각각의 key에 대해서 어떠한 가중치를 가지는 score값을 구하고, Value와 곱해 attention Value를 구함

- $W^0$는 ouput matrix임

## 동작원리

![Untitled](Transforme%203978a/Untitled%2011.png)

- Attetion을 위해 head마다  query, key, value가 필요함
- 각 단어의 Embedding을 이용해 생성 가능
    - 임베딩 차원 ($d_{model}$) → Query, Key, Value 차원 ($d_{model}$)

![Untitled](Transforme%203978a/Untitled%2012.png)

- $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

![Untitled](Transforme%203978a/Untitled%2013.png)

- 실제로는 행렬 곱셈 역산을 한꺼번에 연산이 가능

## Scaled Dot-Producte Attention

![Untitled](Transforme%203978a/Untitled%2014.png)

- Query와 Key값을 내적해 Attention Energies를 구함
- Attention Energies와 Value matrix를 곱하여 attention을 구함

![Untitled](Transforme%203978a/Untitled%2015.png)

- Mask matrix를 이용해 특정 단어를 무시 할 수 있음
- 마스크 값으로 음수 무한 값을 넣어 softmax 함수의 출력이 0%에 가까워지도록 함

![Untitled](Transforme%203978a/Untitled%2016.png)

- $MultiHead(Q,K,V) = Concat(head_1, ...,head_h)W^0$

![Untitled](Transforme%203978a/Untitled%2017.png)

- MultiHead(Q,K,V)를 수행한 뒤에도 차원이 동일하게 유지 됨

## Attention의 종류

![Untitled](Transforme%203978a/Untitled%2018.png)

- Attention에는 3가지 종류의 attention이 사용됨
- Transformer에서는 전부 Multihead attention 사용

- Encoder Self-Attention
    - 각각의 단어가 서로에게 어떤 연관성을 가지는지
    
- Masked Decoder Self-Attention
    - Decoder에서는 전부 수행하는 것이 아니라 앞에 것만 수행

- Encoder-Decoder Attention
    - Query가 Decoder에 있고,  Key,Value는 Encoder에 있음
    - Encoder에서 참조

## Self-Attention

![Untitled](Transforme%203978a/Untitled%2019.png)

- Self-Attention은 Encoder와 Decoder에서 모두 사용
- 매번 입력 문장에서 각 단어가 어떤 단어와 연관성이 높은지를 계산 할 수 있음

## Positional Encoding

![Untitled](Transforme%203978a/Untitled%2020.png)

- Positional Encoding은 다음과 같은 주기 함수를 활용한 공식을 사용
- 각 단어의 상대적인 위치 정보를 네트워크에게 입력

- pos: 단언 번호, i :  각각의 단어에 대한 임베딩 값의 위치
- 따로 학습한 Positional Encoding을 넣어 줄 수 있음

![Untitled](Transforme%203978a/Untitled%2021.png)

# Attention Is All You Need

## Abstract & Introduction

- LSTM과 Gated Recurrent Neural Network 등의 모델이 Sequence Modeling에 있어서 독보적인 성능을 확립하였고, Language Modeling과 Machine Translation 부문에서 큰 발전을 가져왔음
- But, LSTM류의 알고리즘들은 입력값을 순차적으로 받고 입력을 받으면서 매번 hidden state를 갱신하는 과정을 거치기 때문에 병렬화에 어려움이 있음
- 또한, 긴 Sequence 길이에 취약하고, 메모리를 많이 잡아먹는 등의 문제점이 존재
- 이를 효율화하기 위해 수많은 연구들이 진행되었지만, 순차적 입력 방식에 따른 한계점은 근본적으로 해결되기 어려운 양상을 보임
- 본 논문에서는 주로 RNN과 함께 사용되던 Attention Mechanism만을 사용한 새로운 Architecture인 Transformer를 제안하였으며, SOTA를 달성함과 동시에 8개의 P100 GPU로 12시간만에 학습을 이뤄내는등 계산효율성도 크게 향상시킴

## ****Background****

- 기존 연구(Extended Neural GPU, ByteNet, ConvS2S)의 경우 Sequence의 계산을 최대한 줄이는 방향으로 연구를 진행
- CNN을 Basic Building Block으로 사용하여 모든 입력과 출력 위치에 대한 Hidden Representation을 병렬적으로 계산하는 방식을 통해 효율성을 증대
- But, 위와 같은 방식은, Position이 멀어지면 이를 계산하기 위한 추가적인 연산량이 증가하고, ConvS2S의 경우엔 선형적으로 증가하고, ByteNet의 경우 Logarithmical하게 증가함
- 즉, 거리가 멀어질수록 학습이 어려워지게 만듬
- Transformer 방식은 이러한 연산량을 효과적으로 감소시켰으며, 비록 가장 효율적인 Resolution을 찾지 못할 수 있다는 단점도 존재하지만 이는 Multi-Head Attention 방식으로 상쇄할 수 있다고 주장

## ****Model Architecture****

![Untitled](Transforme%203978a/Untitled%2022.png)

### **Encoder**

- 6개의 Stacked Layer로 구성되어 있음
- 각각의 Layer는 2개의 Sub-Layer를 가지는데, 한가지는 Multi-Head Self-Attention Layer이고, 나머지 하나는 Position-Wise Feed Forward Layer
- 각각의 Layer는 ResNet에서 제안됐던 Residual Connection을 적용하였으며 LayerNorm을 적용함
- 임베딩 Layer와 같이 각각의 Layer의 Output의 Dimension은 512

### **Decoder**

- Encoder와 마찬가지로 6개의 Stacked Layer로 구성되어 있음
- 그리고 1개의 Sub Layer가 추가되었는데, 이는 Encoder Stack의 Output을 받아서 Multi-Head Attention을 수행함
- 해당 Layer는 즉, Attention에 필요한 Q,K,V중 Query를 이전 Decoder Layer로 부터 입력받고, Key, Value를 Encoder의 Output 값을 활용
- 첫 Sub Layer에는 Masked Multi-Head Attention을 적용하였는데 이는 Encoder 부분의 현재 시점에서 아직 등장하지 않은 "미래시점"의 단어를 제거해줌으로써 Cheating을 방지하고, 모델이 보다 General하게 학습될 수 있도록 함
- 이는, Transformer가 입력을 순차적으로 받지 않고 한번에 받기 때문인데, i번째 위치에서 예측을 진행할 때에는 i번째 위치 이후에 등장하는 단어들에 대해서는 접근이 불가능하도록 Mask값에 음의 무한대를 취해주는 방법
    
    
    Ex) I Love You. 가 들어오고 이를 독일어로 번역한다고 했을 때, I 만 입력됐을 때, I Love You에 대한 독어 정답값을 미리 알고 있는 Decoder에서 독일어로 I에 해당하는 부분만 남기고 다른 부분을 모두 Masking 해주는 방식
    

## ****Attention****

![Untitled](Transforme%203978a/Untitled%2023.png)

- Attention의 3가지 요소는 Query, Key, Value 임
- I Love You 라는 단어가 있을 때, Query는 Attention을 구하고 싶은 주체

→ 즉 I라는 단어가 I, Love, You라는 단어 각각에 대해서 얼마나 연관성을 가지는지를 구한다고 한다면, Query는 I, Key의 후보군은 I, Love, You가 된다.

- 이렇게 Query와 Key를 통해 구해진 값을 통해 각 단어들간의 연관관계를 파악하고, Softmax를 취해주어 적절한 확률값을 구한뒤, Value와 곱해주어 Weight가 고려된 최종적인 Attention Value를 얻음
- 본 논문에서 사용하는 방식은 Query와 Key 벡터를 행렬곱해주고 Scale을 진행하여 Scaled Dot-Product Attention이라고 명명
- Dot Product Attention과 똑같은 구조이지만, 얻어지는 값에 dimension의 루트값으로 나눠주는 Scale 과정을 거친다는것이 차이점
    
    → 이는 작은 D값을 가질 때는 거의 차이가 없지만, 큰 Dimension값을 가질 때, Softmax가 극도로 작은 Gradient 값을 가지게끔 하는 부작용을 방지하여 성능 향상에 유의미한 효과가 있다고 저자들은 주장
    

## ****Multi-Head Attention****

- Single Attention을 통해 단순히 Attention을 계산하는 것보다, 저자들은 Q,K,V를 h번 다르게 수행하여 병렬적으로 연산하는것이 더 이득이 크다고 주장
- 단일 어텐션을 수행하지 않고, Q,K,V에 대해서 각각 d_k, d_k, d_v 차원으로 변환하는 서로 다른 h개의 Linear Projection을 진행하여 각각의 Projected Value에 대해서 병렬적으로 연산한뒤 concat, linear를 통해 최종 결과 벡터를 얻는 방식
- 이러한 방식은 다양한 Q,K의 조합을 통해서 다양한 Concept으로 학습이 가능하도록 하고 저자들이 주장하듯이, 가장 효율적인 Resolution에 가까운 결과를 얻을 수 있도록 함

## ****Applications of Attention in Transformer****

- 이전 Decoder Layer로 부터 Query를 받아오고, Key와 Value를 Encoder Layer의 최종 output으로 부터 받아오는 방식을 통해 Decoder가 모든 Position에서 입력 문장의 전체를 확인할 수 있도록 하는 구조로, 전형적인 Encoder Decoder Attention 방식을 충실히 따름 (Seq2Seq)
- Encoder는 자체적으로 문장에 대한 Attention을 학습하는 Self Attention Layer를 가지고 있으며, 마찬가지로 이전 Encoder의 Output 값을 받아 이전 영역의 모든 Positon에 대한 정보를 활용할 수 있음
- Decoder도 마찬가지로 Self Attention Layer를 가지고 있으나, Decoder의 Auto Regressive 성질을 보장하기 위해서 leftward information flow(i번째 시점에서, i보다 미래에 등장하는 단어를 미리 조회함으로써 i번째 단어를 결정하는 부적절한 현상)을 막도록 Masking을 진행
- 이는 -inf의 값을 곱해주어 softmax에서 해당 위치의 원소값이 0에 수렴하도록 만드는 방식

## ****Positional Encoding****

- RNN이나 CNN을 전혀 사용하지 않다보니, Sequence에 대한 순서 정보를 전달할 수 없음
- 따라서 저자들은 상대/절대 적인 token의 위치를 전달해주기 위해서 Positional Encoding을 진행
- 학습 가능한 방식, 주기 함수를 통한 고정 방식등 다양하게 사용자들이 선택할 수 있음
- 저자는 Sine, Cosine 함수를 통한 주기함수를 통해서 진행하였으며, 학습을 통한 Positional Encoding 방식과 정확도 차이가 크게 없었다고 함
- 저자들은 성능차이는 그다지 없었지만, 학습에서 미처 접하지 못한 길이의 Sequence가 들어왔을 때도 적절히 처리가 가능한 Sinusoidal 방식을 채택

## Result

![Untitled](Transforme%203978a/Untitled%2024.png)

- 실험 결과 SOTA를 달성하였으며, 학습에 필요한 계산량 또한 획기적으로 줄인 것을 확인할 수 있었다.
- BLEU 벤치마크 뿐만 아니라, Machine Translation, English Constituency Parsing 등 General Task에 대해서도 좋은 결과를 나타냈다.

## ****Conclusion****

- 저자들은 본 논문을 통해 Attention만을 사용한 Architecture인 Transformer를 제안
- Attention 방식과 효율적인 병렬 연산을 통해 계산량을 크게 감소시켰으며, 학습시간을 단축하면서 정확도는 기존 SOTA를 뛰어넘는 결과를 얻어냄 (심지어 여태까지 발표된 모든 방식을 Ensemble한 것 보다도 높은 성능)
- 또한, 특정 Task에 종속적이지 않고 General하게 이곳저곳 적용이 가능하다는 것 또한 실험을 통해 입증

# 참고 자료

- 블로그

[[논문리뷰] Attention is all you need (feat. Transformer)](https://daje0601.tistory.com/287)

[[논문 리뷰] Attention Is All You Need(Transformer)](https://cryptosalamander.tistory.com/162)

[Transformer (Attention is All You Need) 논문 리뷰](https://sanghyu.tistory.com/107)

[[논문 리뷰] Attention is all you need](https://roytravel.tistory.com/107)

[ATTENTION IS ALL YOU NEED 논문 리뷰](https://hipgyung.tistory.com/entry/ATTENTION-IS-ALL-YOU-NEED-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)

[Transformer 논문리뷰 (Attention Is All You Need, NIPS 2017)](https://velog.io/@changdaeoh/Transformer-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0)

- 유튜브

[[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AA621UofTUA)

- 깃허브

[Deep-Learning-Paper-Review-and-Practice/Attention_is_All_You_Need_Tutorial_(German_English).ipynb at master · ndb796/Deep-Learning-Paper-Review-and-Practice](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)