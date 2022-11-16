# BA_03

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/201814625-adf72ae0-4f94-4550-98f1-4986626f82b6.PNG">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Anomaly Detection(이상치 탐지)을 바탕으로 만들어졌습니다.
이번에는 이상치탐지방식의 기본적인 개념과 기법중에서도 Model-based learning에 대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.

## BA_03 Anomaly Detection(이상치 탐지)_개념설명
이상치탐지는 기존 데이터와는 다른 메커니즘으로 생성된 데이터의 샘플입니다. 위 소제목에서는 이상치탐지를 Anomaly Detection이라고 소개했지만 사실 이상치는 **Anomaly,Novelty**로도 사용됩니다. 단어의 의미를 살표보면 **Anomaly**는 약간 부정적 의미를 내포하고 있고 **Novelty**는 긍정적의미를 내포하고있습니다. 이러한 기준은 어떠한 데이터에 적용하나에 따라 다를수 있습니다.예를들면 공장에서는 양품과 불량품이 존재하는데 거기선 불량품이 부정적인 의미를 내포하기에 Anomaly라는 표현이 좀더 적합합니다. 하지만 주식시장에선 갑자기 급등하는 패턴이 분석해야하는 대상이기에 이럴땐 긍정적인 Novelty라는 단어가 알맞습니다.
  
여기까지 글을 읽으신 분은 생각하실겁니다. 이상치탐지는 데이터가 어떤 데이터냐에 따라 긍정의 의미도 부정의의미도 될수 있겠구나 그럼 이상치 데이터는 노이즈 데이터인가?라고 궁금증을 가질수있습니다. 정답은 X입니다. 왜냐하면 노이즈데이터는 무작위성에 기반한 자연현상같은 데이터이고 이상치데이터는 정상데이터를 생성하는 와중에 어떤과정에서 위반되어 생성되는것이기 때문입니다. 

그럼 나아가서 **이상치 탐지문제는 이상치인것과 아닌것을 판단하는것이니 분류문제인가?** 이렇게 질문할수도 있다고 생각합니다. 혹시 이 질문에 대해선 어떻게 생각하시나요?
다음 아래의 그림을 보면서 한번 생각해보면 좋을것같습니다. 
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202180469-7e40b62a-0a72-43e8-b387-399c01b09d9f.png">


- 위 그림을 참고하면 Binary classification 같은경우는 A,B라는 새로운 데이터가 들어오면 양품(o)라고 분류할수있습니다. 
- 하지만 Anomaly detection에서도 새로운 데이터 A,B가 들어오면 불량품(x)라고 해야할까요? 정답은 X입니다. Anomaly detection에서는 양품(o)이 아니다.하고 이야기해야 정확합니다. 

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202186745-6811d21f-ef0e-4c6c-8b24-8364a06c1c45.png">

위 그림을 참고하시면 Classification과 Anomaly detection은 양품데이터가 불량품데이터 보다 상당히 많다는 가정하에서 training방식에서도 차이가 있습니다. Classification은 양품과 불량품 둘다 train 시키지만 Anomaly detection에서는 양품만 train을 시킵니다. 

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/202188297-8a249519-a4dd-4151-afe3-9c9f78d047c6.png">

그럼 언제 Classficiation을 사용해야하고 언제 Anomaly detection을 사용하면 좋을까요? 강필성교수님께서 가르쳐주신내용에 따라 위 그림을 참고하면 좋을듯합니다.
보통은 데이터의 imbalance가 1:99처럼 극심하게 발생 되고 소수범주에 대한 예시들이 충분히 있지 않으면 보통 최후의 보류로써 Anomaly Detection을 사용한다고 합니다. 그러나 데이터의 imblance가 컸어도 소수범주에 대한 예시들이 충분히 있으면 SMOTE와 같은 여러가지 sampling기법을 통해 데이터의 balance를 맞추어 줍니다. 아래에 A,B데이터로 예시를 만들어보았는데 참고하시면 좋을것 같습니다.
||양품의 갯수|불량의 갯수|기법|
|:---:|:---:|:---:|:---:|
| A |999,000|1,000|**Classification**|
| B |9,990|10|**Anomaly detection**|

