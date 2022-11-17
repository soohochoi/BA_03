# BA_03 Anomaly Detection(이상치 탐지)

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/201814625-adf72ae0-4f94-4550-98f1-4986626f82b6.PNG">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Anomaly Detection(이상치 탐지)을 바탕으로 만들어졌습니다.
이번에는 이상치탐지방식의 기본적인 개념과 기법중에서도 Model-based learning에 대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.

## BA_03 Anomaly Detection(이상치 탐지)_개념설명
이상치탐지는 기존 데이터와는 다른 메커니즘으로 생성된 데이터의 샘플입니다. 위 소제목에서는 이상치탐지를 Anomaly Detection이라고 소개했지만 사실 이상치는 **Anomaly,Novelty**로도 사용됩니다. 단어의 의미를 살표보면 **Anomaly**는 약간 부정적 의미를 내포하고 있고 **Novelty**는 긍정적의미를 내포하고있습니다. 이러한 기준은 어떠한 데이터에 적용하나에 따라 다를수 있습니다.예를들면 공장에서는 양품과 불량품이 존재하는데 거기선 불량품이 부정적인 의미를 내포하기에 Anomaly라는 표현이 좀더 적합합니다. 하지만 주식시장에선 갑자기 급등하는 패턴이 분석해야하는 대상이기에 이럴땐 긍정적인 Novelty라는 단어가 알맞습니다.
  
여기까지 글을 읽으신 분은 생각하실겁니다. 이상치탐지는 데이터가 어떤 데이터냐에 따라 긍정의 의미도 부정의의미도 될수 있겠구나 그럼 이상치 데이터는 노이즈 데이터인가?라고 궁금증을 가질수있습니다. 정답은 X입니다. 왜냐하면 노이즈데이터는 무작위성에 기반한 자연현상같은 데이터이고 이상치데이터는 정상데이터를 생성하는 와중에 어떤과정에서 위반되어 생성되는것이기 때문입니다. 아래에 그림을 보시면 조금 더 이해하기에 수월하실겁니다. (a)그림에 A점은 다른데이터들과 떨어져서 분포하기에 정상데이터를 생성하는 와중에 어떤과정에서 위반되어 생성되어진 이상치데이터이고 (b)그림에 A점은 명확하게 다른데이터들과 떨어져서 분포한다고 말할수 없기에 무작위성에 기반한 자연현상같은 노이즈데이터라고 말합니다.
  
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202194818-fcf0b499-5123-42d7-ba6a-9d178511cbca.png">

그럼 나아가서 **이상치 탐지문제는 이상치인것과 아닌것을 판단하는것이니 분류문제인가?** 이렇게 질문할수도 있다고 생각합니다. 혹시 이 질문에 대해선 어떻게 생각하시나요?
다음 아래의 그림을 보면서 한번 생각해보면 좋을것같습니다. 
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202180469-7e40b62a-0a72-43e8-b387-399c01b09d9f.png">


- 위 그림을 참고하면 Binary classification 같은경우는 A,B라는 새로운 데이터가 들어오면 양품(o)라고 분류할수있습니다. 
- 하지만 Anomaly detection에서도 새로운 데이터 A,B가 들어오면 불량품(x)라고 해야할까요? 정답은 X입니다. Anomaly detection에서는 양품(o)이 아니다.하고 이야기해야 정확합니다. 

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202186745-6811d21f-ef0e-4c6c-8b24-8364a06c1c45.png">

- 위 그림을 참고하시면 Classification과 Anomaly detection은 양품데이터가 불량품데이터 보다 상당히 많다는 가정하에서 training방식에서도 차이가 있습니다. Classification은 양품과 불량품 둘다 train 시키지만 Anomaly detection에서는 양품만 train을 시킵니다. 

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202188297-8a249519-a4dd-4151-afe3-9c9f78d047c6.png">

그럼 언제 Classficiation을 사용해야하고 언제 Anomaly detection을 사용하면 좋을까요? 강필성교수님께서 가르쳐주신내용에 따라 위 그림을 참고하면 좋을듯합니다.
보통은 데이터의 imbalance가 1:99처럼 극심하게 발생 되고 소수범주에 대한 예시들이 충분히 있지 않으면 보통 최후의 보류로써 Anomaly detection을 사용한다고 합니다. 그러나 데이터의 imblance가 컸어도 소수범주에 대한 예시들이 충분히 있으면 SMOTE와 같은 여러가지 sampling기법을 통해 데이터의 balance를 맞추어 줍니다. 아래에 A,B데이터로 예시를 만들어보았는데 참고하시면 좋을것 같습니다.
||양품의 갯수|불량의 갯수|기법|
|:---:|:---:|:---:|:---:|
| A |999,000|1,000|**Classification**|
| B |9,990|10|**Anomaly detection**|

## BA_03 Anomaly Detection(이상치 탐지)_Model_based_learning
이상치 탐지가 어떤것인지 아셨나요? 그럼 이번에는 모델을 통한 이상치탐지기법을 알아보려고합니다. 

- ### Auto encoder(AE)
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/202328901-561ae8f2-141d-48e6-b2d8-46e9bb843943.png">
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202329709-f4b95401-8be7-4202-87bf-c33de8366391.png">

오토인코더(AE)는 입력층과 출력이 동일한 인공 신경망 구조입니다. 단 입력층과 출력층의 차원은 동일하지만 은닉층은 입력층의 차원의 수를 넘어 설수 없습니다. 쉽게 이야기하여서 콩심은데 콩나고 팥심은데 팥이나는 구조입니다. 그럼 궁금중이 생기실텐데 왜 콩을 넣어서 콩이 나오는 모델을 왜 사용할까요? 이유는 크게 2가지가 있습니다. 
1. 차원축소의 목적으로 AE를 학습시켜 성능을 확인 한 뒤 잠재 벡터인 feature를 다른 기계학습모형의 인풋으로 사용함
2. 입력정보와 AE출력 정보간 차이를 이용한 분석을 통해 이상치를 분석함

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202330000-5d9bb67a-f862-4452-9c15-df7703a783fc.png">

AE의 구조는 Loss function은 Reproduction된 출력 값에서 입력 값을 뺀 값으로 이루어지며 anomaly score라고 명명되어집니다. 위에 그림을 참고하시면 아까전에 이야기한것처럼 h(x)는 𝑥 ̂  와 𝑥의 차원보다 크지 않아야 합니다.  즉, h(x)는 중요한 입력 값의 정보를 축약 해야합니다.  그러면 위 그림에 without bottle layer처럼 h(x)의 차원 = 𝑥 의 차원과 같으면 어떻게 될까요? without bottle layer처럼 입력층(=출력층)과  입력정보와 출력정보를 히든레이어의 수와 같게 설정을 하면 입력정보와 출력정보를 그대로 외워 버리기에 overfitting이 발생됩니다. 따라서 With bottle layer처럼 입력 정보와 출력정보보다 히든레이어수의 차원이 작아야 합니다. 
  
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202331854-8ddfd278-c22f-49c2-8ae8-bb54ea91036f.png">

이처럼 AE는 중요 feature만을 압축하기에 용량도 작고 품질도 더 좋습니다. 또한 차원의 저주를 예방 할 수 있습니다. 뿐만 아니라 위그림의 기여도를 보시면 복원이 잘 되지 않을 경우, 기여도에 대한 차이도 알수 있습니다. AE또한 단점이 존재합니다. 입력에 대한 약간의 변형에도 모델이 민감하게 반응합니다. 따라서 단점을 보안하기위해 입력에 noise를 첨가해 noise가 제거된 결과값이 나오도록 합니다. 이 과정은 모델을 더욱 robust하게 만들도록 보완합니다. Noise는 보통 Random Gaussian noise가 사용됩니다. 

## BA_03 Anomaly Detection(이상치 탐지)_Model_based_learning Auto encoder 실습코드

- ### 데이터 설명
이번에 사용되는 데이터는 Kaggle에서 제공하는 Credit Card Fraud Dectection입니다. 이 데이터는 2013년 9월에 신용카드 사용자들의 실제 거래기록으로 총 284,807 건의 거래내역이 제공됩니다. 이 중 정상거래(Normal Transaction)는 284,315건이고 492건이 사기 거래(Fraud Transaction)입니다. 사기 거래가 전체거래에 0.172% 차지하므로 위에서 살펴본 내용처럼 매우 imbalance한 특징을 가진 데이터셋입니다.

[Credit Card Fraud Dectection 다운로드](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


데이터는 비
