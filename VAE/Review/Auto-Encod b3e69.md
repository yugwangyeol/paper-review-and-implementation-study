# Auto-Encoding Variational Bayes(VAE)

https://www.notion.so/Auto-Encoding-Variational-Bayes-VAE-9f340fee45a1497998e23f771e69737b

# 0.Abstract

- 큰 데이터셋과 계산이 불가능한 posterior 분포를 가지는 연속형 잠재 변수를 가지고 있을 때, 어떻게 directed probabilistic model을 효율적으로 학습하고 추론할 수 있을까?
- 큰 데이터셋에도 확장할 수 있고 가벼운 미분가능성 조건이 있다면 계산이 불가능한 경우에도 작동하는 stochastic variational inference and learning 알고리즘을 제안

- 논문에서 기여하는 두 가지
    
    
    1. variational lower bound의 reparameterization이 표준적인 stochastic gradient 방법론들을 사용하여 직접적으로 최적화될 수 있는 lower bound estimator를 만들어낸다는 것을 보임
    2. 각 datapoint가 연속형 잠재 변수를 가지는 i.i.d. 데이터셋에 대해서, 제안된 lower bound estimator를 사용해 approximate inference model(또는 recognition model이라고 불림)을 계산이 불가능한 posterior에 fitting 시킴으로써 posterior inference가 특히 효율적으로 만들어질 수 있다는 점을 보임. 실험 결과에 이론적 이점이 반영됨
    

# 1.Intorduction

- 연속적인 잠재 변수 및/또는 매개 변수가 계산하기 어려운 사후 분포를 갖는 directed probabilistic model을 가지고 어떻게 하면 효율적인 approximate inference와 learning을 수행할 수 있을까?

- Variational Bayesian (VB) 접근법은 계산 불가능한 posterior로의 근사의 최적화를 포함
- 불행하게도, 일반적인 mean-field 접근법은 근사적인 posterior에 대한 기댓값의 분석적 해를 요구하며, 일반적인 경우에 이 또한 계산 불가능
- 논문에는 variational lower bound의 reparameterization이 어떻게 lower bound의 미분 가능한 unbiased estimator를 만들어내는지를 보임

- 이 SGVB (Stochastic Gradient Variational Bayes) estimator는 연속형 잠재 변수나 파라미터를 가지는 어떤 모델에서도 효율적인 approximate posterior inference를 위해 사용될 수 있으며, 표준 gradient ascent 기법을 사용해서 직접적으로 최적화함
    - approximate posterior inference란, posterior를 직접적으로 계산할 수 없기 때문에 근사적인 방법으로 구하는 것을 의미합니다. 딥러닝에서의 inference와는 차이가 있음
    
- 각 datapoint가 연속형 잠재 변수를 가지는 i.i.d. 데이터셋의 경우에, 우리는 Auto-Encoding VB (AEVB) 알고리즘을 제안

- AEVB 알고리즘에서, 우리는 단순한 ancestral sampling을 사용하여 매우 효율적인 approximate posterior inference를 수행하게 해주는 recognition model을 최적화하기 위해 SGVB estimator를 사용함으로써 추론과 학습을 특히 효율적으로 만들 수 있으며, 이는 각 datapoint에 MCMC와 같은 expensive 한 반복적 추론 방법 없이도 모델의 파라미터들을 효율적으로 학습할 수 있도록 함
    - 여기서 expensive 하다는 의미는 연산 관점에서 연산량이 많다는 의미
    
- 학습된 approximate posterior inference model은 recognition, denoising, representation, visualization와 같은 목적을 위해 사용될 수 있음
- Recognition model에 neural network가 사용되었을 때, 우리는 variational auto-encoder에 도달

# 2.Method

- 이번 section에서의 전략은 연속형 잠재 변수를 가지고 있는 다양한 directed graphical model을 위한 lower bound estimator (a stochastic objective function)을 도출하는 데 사용
- 우리는 각 datapoint가 잠재 변수를 가지는 i.i.d 데이터셋이 존재하는 일반적인 경우로만 제한할 것이며, 파라미터들에 대해서는 maximum likelihood (ML)이나 maximum posteriori (MAP) 추론을 진행하고 잠재 변수에 대해서는 variational inference를 수행
    
    
    ex) 이 시나리오를 우리가 variational inference를 global parameters에 수행하는 경우로 확장하는 것은 간단하며, 이는 appendix에 들어있고 이 케이스에 해당하는 실험은 future work로 남김
    
- 논문의 방법은 streaming data와 같은 online- non-stationary setting에도 적용할 수 있으나, 단순함을 위해 고정된 데이터셋을 가정
    
    
    ![Auto-Encod%20b3e69/Untitled.png](Auto-Encod%20b3e69/Untitled.png)
    
    - Figure1; 우리가 고려하는 directed grphical model의 유형을 나타냄
    - 실선은 generative model $p_\theta(z)p_\theta(x|z)$이며, 점선은 계산 불가능한 posterior$p_\theta(z|x)$로의 variational appoximation $q_\Phi(z|x)$를 나타낸다. variational parameters $\Phi$는 generative model parameters $\theta$와 함께 학습됨.
    

## 2.1 Problem Scenario

- 연속형 변수, 이산형 변수 x의 N개의 i,i,d. smaple로 구성된 데이터 셋 X = {X^{(i)}}^N_{i=1}을 고려
- 관측되지 않은 연속형 랜덤 변수 z를 포함하는 어떤 random process에 의해서 데이터가 생성되었다고 가정
- Process는 2개의 step으로 구성된다.
    
    
    1. z^{(i)}는 어떤 사전 분포 p_{\theta^*}(z)로부터 생성됨
    2. x^{(i)}는 어떤 조건부 분포 p_{\theta^*}(x|z)로부터 생성됨
    
- 논문에서는 prior $p_{\theta^*}(z)$와 likelihood  $p_{\theta^*}(x|z)$가 parametric famillies od distribution $p_{\theta^*}(z)$와 $p_{\theta^*}(x|z)$로부터 왔다고 사정하며, 그들의 PDF는 $\theta$와 $z$에 대해서 거의 모든 곳에서 미분 가능하다고 가정
- 불행하게도, 우리의 관점에서 이 과정의 많은 것들이 숨겨져 있다: 즉, true parameters θ∗와 잠재 변수들의 값 z(i)은 우리에게 알려져 있지 않다.
- 매우 중요하게, 우리는 주변 확률이나 사후 확률에 대해서 일반적인 단순화한 가정을 만들지 않는다.

- 대조적으로, 우리는 이러한 경우에서도 효율적으로 작동하는 일반적인 알고리즘에 관심이 있다
    
    
    1. Intractability: marginal likelihood의 적분 $p_\theta(x) = \int p_\theta(z)p_\theta(x|z)dz$이 계산 불가능한 경우(따라서 우리는 marginal likelihood를 평가하거나 미분할 수 없다)를 말하며, true posterior density $p_\theta(z|x) = p_\theta(x|z)p_\theta(z)/p_\theta(x)$도 계산 불가능하며(따라서 EM algorithm을 사용할 수 없음), 어떤 합리적인 mean-field VB algorithm을 위해 요구되는 적분도 계산 불가능한 경우를 말함.
        
        
        1. 이러한 계산 불가능성은 꽤 흔하고 nonlinear hidden layer를 포함하는 신경망과 같은 적당히 복잡한 likelihood function $p_\theta(x|z)$ 의 경우에서 나타난다. 
        2. 즉, posterior를 적분과 같은 계산을 통해서 구할 수 없는 경우에도 알고리즘이 작동해야 한다는 말입니다. marginal likelihood의 적분식을 보면, z에 대해서 적분을 시행하게 되는데 우리는 z에 대해서 알지 못하기 때문에 이를 실제 적분을 시행할 수 없다.
        
    2. A large dataset: 우리는 너무 많은 데이터를 가지고 있어서 batch optimization은 너무 연산량이 많다. 우리는 small minibatch나 심지어 하나의 datapoints만 사용해서 parameter update를 하고 싶다.
    
- 위의 시나리오에서 3개의 관련된 문제들에 관심이 있으며, 이에 대한 해결책을 제안
    
    
    1. Parameters θ에 대한 Efficient approximate ML or MAP 추정. 예를 들어서, natural process를 분석하는 경우와 같이 매개 변수 자체에 관심이 있을 수 있다. 그들은 또한 우리가 숨겨진 random process를 모방하도록 해주며 real data를 닮은 인공 데이터를 생성하게 해 준다.
    
    1. Parameter θ의 선택을 위한 관측된 값 x가 주어졌을 때 잠재 변수 z에 대한 효율적인 approximate posterior inference. 이는 coding이나 data representation task에 유용
    
    1. 변수 x의 효율적인 approximate marginal inference. 이는 우리가 x에 대한 prior가 요구되는 모든 종류의 inference task를 수행할 수 있도록 한다. computer vision에서의 일반적인 응용 사례는 image denoising, inpainting, super-resolution을 포함

- 위의 문제들을 해결하기 위해, recognition model $q_\Phi(z|x)$를 도입한다. 이는 계산 불가능한 true posterior $p_\theta(z|x)$에 대한 추정
- Mean-field variational inference에서의 approximate posterior와는 대조적으로, 이는 반드시 factorial일 필요는 없으며 parameters $\Phi$는 closed-form expectation으로부터 계산되지 않음을 유의한다.
- 그 대신에, 우리는 생성 모델 파라미터 $\theta$와 recognition model 파라미터 $\Phi$를 동시에 학습하는 방법을 도입, Coding theory 관점에서, 관측되지 않은 변수 z는 latent representation혹은 code라는 해석을 가지고 있음
- 이번 논문에서, 우리는 그러므로 recognition model $q_\Phi(z|x)$를 확률적 encoder로 지칭하며, 이는 datapoint x가 주어졌을 때 이 모델이 datapoint x가 생성될 수 있는 지점인 code z의 가능한 값들에 대한 분포(예를 들어, Gaussian 분포)를 만들어 내기 때문이다.
- 유사한 방식으로, 우리는 $p_\theta(x|z)$를 확률적 decoder로 지칭할 것이며, code z가 주어졌을 때 이는 가능한 x의 값들에 대한 분포를 만들어냄

![Auto-Encod%20b3e69/Untitled%201.png](Auto-Encod%20b3e69/Untitled%201.png)

## 2.2 The Variaional Bound

- Marginal likelihood는 각각의 datapoint들의 marginal likelihood의 합으로 구성된다.
- 즉,$logp_\theta(x^{(1)},...,x^{(N)}) = \sum^N_{i=1}logp_\theta(x^{(i)})$ 이며, 이는 다음과 같이 다시 쓰일 수 있음.

- 우변의 첫 번째 항은 true posterior와의 approximate 간의 KL divergence이다. (KL divergence란, GAN paper에서도 봤지만 두 분포의 차이를 계산하는 방법입니다. 위의 식에서 true posterior가 $p_\theta(z|x^{(i)})$를 의미하고, approximate가 $q_\theta(z|x^{(i)})$를 의미함. approximate는 위에서 Recognition model, 혹은 확률적 Encoder라는 이름으로도 설명됨. )

![Auto-Encod%20b3e69/Untitled%202.png](Auto-Encod%20b3e69/Untitled%202.png)

- KL-divergence가 비음(non-negative) 이므로, 우변의 두 번째 항 $L(\theta,\Phi;x^{(i)})$는 datapoint i의 대한 marginal likelihood에 대한 (variational) lower bound이다. 이는 다음과 같이 쓰일 수 있다.

![Auto-Encod%20b3e69/Untitled%203.png](Auto-Encod%20b3e69/Untitled%203.png)

![Auto-Encod%20b3e69/Untitled%204.png](Auto-Encod%20b3e69/Untitled%204.png)

- 우리는 variational parameters $\Phi$와 generative parameters $\theta$둘 다에 대해서 lower bound $L(\theta,\Phi;x^{(i)})$를 미분하고 최적화하고 싶다.
- 하지만, lower bound의 $\Phi$에 대한 gradient는 다소 문제가 있다.
- 이런 종류의 문제에 대한 일반적인 Monte Carlo gradient estimator는

![Auto-Encod%20b3e69/Untitled%205.png](Auto-Encod%20b3e69/Untitled%205.png)

이며,$z^{(i)}$ ~ $q_\Phi(z|x^{(i)})$이다

- 이러한 gradient estimator는 매우 높은 variance를 보이고, 우리의 목적에는 실용적이지 않다.

## 2.3 The SGVB Estimator And AEVB Algorithm

- 이번 section에서 우리는 the lower bound와 파라미터에 대한 lower bound의 미분 값에 대한 실용적 estimator를 소개한다.
- approximate posterior를 $q_\Phi(z|x)$의 형태로 가정하지만, 이 기술은 $q_\Phi(z)$의 경우에도 적용될 수 있음
- 즉, x에 대한 조건부가 아닐 때도 가능하다는 의미이다.
- 주어진 parameters에 대한 posterior를 추론하는 fully variational Bayesian method는 appendix에 쓰여있다.
- Section 2.4에서 서술된 어떤 가벼운 조건 하에서 주어진 approximate posterior $q_\Phi(z|x)$에 대해 우리는 random variable $z$ ~ $q_\Phi(z|x)$를 (보조) noise variable ϵ의 미분 가능한 transformation $g_\Phi(\epsilon,x)$를 사용하여 reparameterize 할 수 있다.

![Auto-Encod%20b3e69/Untitled%206.png](Auto-Encod%20b3e69/Untitled%206.png)

- 적절한 distribution $p(\epsilon)$과 function $g_\Phi(\epsilon,x)$를 선택하는 것에 대한 일반적인 전략은 section 2.4에서 확인할 수 있다.
- 우리는 이제 $q_\Phi(\epsilon,x)$에 대한 어떠한 함수 $f(z)$의 기댓값에 대한 Monte Carlo estimate를 구성할 수 있다. (기댓값을 직접 구하기 어렵기 때문에, Monte Carlo 기법을 활용해서 추정하는 것이라고 보면 될 것 같습니다.)

![Auto-Encod%20b3e69/Untitled%207.png](Auto-Encod%20b3e69/Untitled%207.png)

- 우리는 이 기술을 variational lower bound (식 (2))에 적용하여 이를 통해 generic Stochastic Gradient Variational Bayes (SGVB) estimator $\overset{\sim}{L} ^A(\theta,\Phi;x^{(i)}) \simeq L(\theta,\Phi;x^{(i)})$를 만들 수 있다.
- 논문에 쓰여있는 그대로, Monte Carlo estimate를 식 (2)에 적용하면 식 (6)을 도출할 수 있음

![Auto-Encod%20b3e69/Untitled%208.png](Auto-Encod%20b3e69/Untitled%208.png)

- 종종, 식 (3)의 KL-divergence $D_{KL}(q_\phi(z|x^{(i)})||p_\theta(z))$는 분석적으로 통합될 수 있으며(appendix B를 참조), expected reconstruction error $E_{q\Phi(z|x^{(i)})}[logp_\theta(x^{(i)}|z)]$는 샘플링에 의한 추정이 요구된다.
- KL-divergence는 $\Phi$를 규제하는 것으로 해석될 수 있으며, approximate posterior가 prior $p_\theta(z)$에 가까워지도록 유도한다.
- 이는 두 번째 버전의 SGVB estimator $\overset{\sim}{L} ^B(\theta,\Phi;x^{(i)}) \simeq L(\theta,\Phi;x^{(i)})$를 만들어내며, 이는 식 (3)에 대응되고, generic estimator(식 (6) 번)에 비해서 일반적으로 더 낮은 variance를 가진다.

![Auto-Encod%20b3e69/Untitled%209.png](Auto-Encod%20b3e69/Untitled%209.png)

- Ndata point를 가지는 dataset X로부터 여러 개의 datapoints가 주어질 때, 미니 배치를 기반으로 전체 데이터셋의 marginal likelihood lower bound의 estimator를 만들 수 있다.

![Auto-Encod%20b3e69/Untitled%2010.png](Auto-Encod%20b3e69/Untitled%2010.png)

- Minibatch $X^M = \{X^{(i)}\}^M_{i=1}$은 N data point를 가지는 전체 데이터셋 X로부터 M개의 datapoints가 랜덤 하게 뽑힌 것이다.
- 우리 실험에서, datapoint 당 샘플의 수인 LL은 1로 지정할 수 있으며, 그동안 minibatch size MM은 충분히 커야 한다. 예를 들어, MM = 100 정도는 되어야 한다.
- 미분 값 $\nabla_{\theta,\Phi}\overset{\sim}{L}(\theta;X^M)$는 얻어질 수 있고, 결과로 나오는 gradient는 SGD나 Adagrad와 같은 stochastic optimization emthod와 결합되어 사용될 수 있다.
- Stochastic gradient를 계산하기 위한 기초적인 접근법이 algorithm 1에 나와있다.

![Auto-Encod%20b3e69/Untitled%2011.png](Auto-Encod%20b3e69/Untitled%2011.png)

![Auto-Encod%20b3e69/Untitled%2012.png](Auto-Encod%20b3e69/Untitled%2012.png)

- Auto-encoders와의 연결은 식 (7)에 나타난 objective function을 봤을 때 명확해진다.
- prior로부터 approximate posterior와의 KL divergence인 첫 번째 항은 regularizer의 역할을 하게 되며, 두 번째 항은 expected negative reconstruction error이다.
- 함수 $g_\Phi(.)$은 datapoint x(i)와 random noise vector ϵ(l)를 해당 datapoint에 대한 approximate posterior로부터 나온 샘플로 맵핑을 한다.
- 즉, $z^{(i,l)}=g_\Phi(\epsilon^{(i)},x^{(i)})$이고 $z^{(i,l)} \sim q_\Phi(z|x^{(i)})$이다.
- 그 결과로, sample $z^{(i,l)}$는 함수 $logp_\theta(x^{(i)}|z^{(i,l)})$의 input이 되며, 이는 $z^{(i,l)}$가 주어졌을 때, generative model 하에서 datapoint $x^{(i,l)}$의 확률 밀도(혹은 질량)와 같다. 이 항은 auto-encoder에서 negative reconstruction error이다.

## 2.4 The reparameterzation Trick

- 문제를 해결하기 위해, $q_\Phi(z|x)$로부터 샘플을 생성해내기 위한 대안적인 방법을 적용
- 본질적인 parameterization trick은 꽤 단순
- z가 연속적인 랜덤 변수라고 하고, $z \sim q_\Phi(z|x)$는 어떤 조건부 분포라 하자
- 그러고 나서 랜덤 변수 z를 deterministic 변수 $z = g_\Phi(z|x)$로 표현하는 것이 가능하며, ϵ은 독립적인 주변 분포 p(ϵ)에서 나온 보조 변수이며, $g_\Phi(.)$은 ϕ에 의해서 parameterized 된 vector-valued 함수이다.
- 이러한 reparameterization은 $g_\Phi(z|x)$에 대한 기댓값을 재작성하는 데 사용될 수 있기 때문에 우리의 경우에 유용하며, 이러한 기댓값의 Monte Carlo estimate는 $\Phi$와 관련하여 미분 가능하게 될 수 있다.
- 증명은 다음과 같다.
    
    
    - Deterministic mapping $z = g_\Phi(z|x)$가 주어졌을 때, $q_\Phi(z|x)\prod_idz_i = p(\epsilon)\prod_id\epsilon_i$라는 사실을 알고 있다.
    - 그러므로, $\int q_\phi(z|x)f(z0dz = \int p(\epsilon)f(z)d\epsilon = \int p(\epsilon)f(g_\phi(x,epsilon^{(i)}))d\epsilon$이다.
    - 예를 들면
        
        
        - univariate Gaussian case를 생각해보면, $z \sim p(z|x) = N(\mu,\sigma^2)$이다.
        - 이 경우에서는, 유효한 reparameterization은 $z = \mu + \sigma \epsilon$이며, ϵ은 ϵ∼N(0,1)을 만족하는 보조 노이즈 변수이다.
    - 그러므로
    
    ![Auto-Encod%20b3e69/Untitled%2013.png](Auto-Encod%20b3e69/Untitled%2013.png)
    
    를 만족한다.
    
- 어떤 $q_\phi(z|x)$에 대해서, 우리는 미분 가능한 transformation $g_\phi(.)$과 보조 변수 ϵ∼p(ϵ)을 선택할 수 있을까? 3가지 기초적인 접근법은 다음과 같다.
    
    
    1. 계산 가능한 inverse CDF. 이 경우에는, $\epsilon \sim \mu(0,1)$라 하고, $g_\phi(\epsilon,x)$를 $q_\phi(\epsilon,x)$의 inverse CDF라고 하자.
        1. 예시: Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Reciprocal, Gompertz, Gumbel and Erlang 분포가 될 수 있다.
        
    2. Gaussian 예시와 유사하게, 어떠한 "location-salce" 계통의 분포를 보조 변수 ϵ의  standard distribution(location =0, scale = 1)으로 선택할 수 있으며, $g(.)$= location + scale ⋅ϵ로 놓는다. 
        1. 예시: Laplace, Elliptical, Student's t, Logistic, Uniform, Triangular and Gaussian distribution.
        
    3. 랜덤 변수를 보조 변수들의 다른 transformation으로 표현할 수 있다. 예시: Log-Normal(normal하게 분포된 변수의 exponentiation), Gamma(exponentially 하게 분포된 변수들에 대한 합), Dirichlet(Gamma variates의 가중 합), Beta, Chi-Squared, and F 분포.
    
    1. 모든 세 접근법이 실패했을 때, inverse CDF에 대한 좋은 근사가 존재하면 PDF에 비교될 정도의 시간 복잡도가 요구된다.

![Auto-Encod%20b3e69/Untitled%2014.png](Auto-Encod%20b3e69/Untitled%2014.png)

# 3. Example: Variational Auto-Encoder

- 이번 section에서 확률적 encoder $q_\phi(z|x)$(generative model $q_\phi(z|x)$의 posterior에 대한 근사)로 신경망을 사용했을 때의 예시를 보여주며, parameter $\phi$와 $\theta$가 AEVB algorithm을 가지고 동시에 최적화되는 예시를 보여준다.
- 잠재 변수에 대한 prior를 centered isotropic multivariate Gaussian $p_\theta(z) = N(z;0,1)$라고 하자.  이런 경우에 prior는 parameter가 없다는 것을 알아두자.

- 우리는 $p_\theta(x|z)$를 실수 데이터의 경우에 multivariate Gaussian으로, binary data의 경우 Bernoulli로 놓으며 이 분포의 파라미터들은 MLP(하나의 hidden layer를 가지는 fully-connected 신경망)를 가지고 z로부터 계산될 수 있다.
- True posterior $p_\theta(z|x)$는 이 경우에 계산이 불가능하다는 것을 알아두자.
- 반면에, $q_\phi(z|x)$의 형태에는 많은 자유가 있으며, 우리는 true (but intractable) posterior가 approximately diagonal covariance를 가지는 approximate Gaussian을 맡는다고 가정한다.
- 이 경우에, 우리는 variational approximate posterior를 diagonal covariance structure를 가지는 multivariate Guassian으로 정할 수 있다.

![Auto-Encod%20b3e69/Untitled%2015.png](Auto-Encod%20b3e69/Untitled%2015.png)

- Approximate posterior의 평균과 표준편차인 $\mu(i)$와 $\sigma(i)$는 datapoint x($x(i)$와 variational parameter $\phi$을 가지는 비선형 함수인 encoding MLP의 output이다. (appendix C 참조)
- Section 2.4에서 설명했던 대로, 우리는 $\epsilon^{(l)} \sim N(0,l)$일 때 $z^{(i,l)}=g_\phi(x^{(i)},\sigma^{(i)})=\mu^{(i)}+\sigma^{(i)} \bigodot \epsilon^{(l)}$을 사용하여 $z^{(i,l)} \sim q_\phi(z|x^{(i)})$로부터 샘플링을 진행한다.
    - $\bigodot$는 element-wise product를 나타낸다.
    
- 이 모델에서 $p_\theta(z)$(the prior)와 $q_\phi(z|x)$는 둘 다 Gaussian이다; 이 경우에, 우리는 식 (7)의 estimator를 사용할 수 있으며 추정 없이 KL divergence가 계산되고 미분될 수 있다.
- 모델에 대한 그 결과로의 estimator와 datapoint $X^{(i)}$는 다음과 같다.

![Auto-Encod%20b3e69/Untitled%2016.png](Auto-Encod%20b3e69/Untitled%2016.png)

![Auto-Encod%20b3e69/Untitled%2017.png](Auto-Encod%20b3e69/Untitled%2017.png)

![Auto-Encod%20b3e69/Untitled%2018.png](Auto-Encod%20b3e69/Untitled%2018.png)

- appendix C와 위에서 설명한 대로, decoding term $logp_\theta(x^{(i)}|z^{(i,l)})$는 Bernoulli나 Gaussian MLP이며, 우리가 모델링하려는 데이터의 유형에 의존한다.

# 4.Related Work

- 생략

# 5.Experiment

- MNIST 와 Frey face dataset 을 가지고 generative model을 학습시켰다고함

![Auto-Encod%20b3e69/Untitled%2019.png](Auto-Encod%20b3e69/Untitled%2019.png)

# 6.Conclusion

- 우리는 연속형 잠재 변수를 효율적으로 approximate inference 하기 위한 새로운 variational lower bound의 estimator인 Stochastic Gradient VB (SGVB)를 소개했다. 제안된 estimator는 단도직입적으로 미분될 수 있고 standard stochastic gradient method를 사용해 최적화될 수 있다.

- 각 datapoint가 연속형 잠재 변수를 가지는 i.i.d dataset의 경우에, 우리는 효율적인 추론과 학습이 가능한 효율적인 알고리즘인 Auto-Encoding VB (AEVB)를 소개했으며, SGVB estimator를 사용하여 approximate inference model을 학습한다.

# 7. Future Work

1. AEVB와 공동으로 학습한 인코더와 디코더에 사용되는 neural network (예: convolutional networks)을 가진 계층적 생성 architecture를 학습하는 것

1. dynamic Bayesian networks와 같은 시계열 모델

1. SGVB의 global parameters로의 응용

1. 잠재 변수를 이용한 supervised models, 이는 복잡한 노이즈 분포를 학습하는데 유용하다.

# 8. Reference

리뷰
[https://judy-son.tistory.com/11](https://judy-son.tistory.com/11)

[https://kvfrans.com/variational-autoencoders-explained/](https://kvfrans.com/variational-autoencoders-explained/) [http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html](http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html) [http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-2. html](http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-2.html)[http://jaejunyoo.blogspot.com/2017/05/auto-encoding-variational-bayes-vae-3. html](http://jaejunyoo.blogspot.com/2017/05/auto-encoding-variational-bayes-vae-3.html)[https://velog.io/@changdaeoh/vaereview](https://velog.io/@changdaeoh/vaereview)

[https://wingnim.tistory.com/70](https://wingnim.tistory.com/70) 
[https://cumulu-s.tistory.com/24](https://cumulu-s.tistory.com/24)

유튜브
[https://www.youtube.com/watch?v=KYA-GEhObIs](https://www.youtube.com/watch?v=KYA-GEhObIs)

코드
[https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb](https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb) [https://github.com/hwalsuklee/tensorflow-mnist-VAE](https://github.com/hwalsuklee/tensorflow-mnist-VAE)
