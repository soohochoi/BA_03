# BA_03

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/201814625-adf72ae0-4f94-4550-98f1-4986626f82b6.PNG">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Anomaly Detection(이상치 탐지)을 바탕으로 만들어졌습니다.
이번에는 이상치탐지방식의 기본적인 개념과 기법중에서도 Model-based learning에 대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.

## BA_03 Anomaly Detection(이상치 탐지)_개념설명
위 소제목에서는 이상치탐지를 Anomaly Detection이라고 소개했지만 사실 이상치는 **Anomaly,Novelty**로도 사용됩니다. 단어의 뜻을 살피면 둘다 이상치이지만 Anomaly는 약간 부정적 의미를 내포하고 있고 Novelty는 긍정적의미를 내포하고있습니다. 이러한 기준은 어떠한 데이터에 적용하나에 따라 다를수 있습니다. 일반적으로 공장에서는 양품과 불량품이 존재하는데 거기선 불량품이 부정적인 의미를 내포하기에 Anomaly라는 표현이 좀더 적합입니다. 하지만 주식시장에선 갑자기 급등하는 패턴이 분석해야하는 대상이기에 이럴땐 긍정적인 Novelty라는 단어가 알맞습니다.
여기까지 글을 읽으신 분은 생각하실겁니다. 이상치탐지는 데이터가 어떤 데이터냐에 따라 달라지겠구나...근데 그럼 outlier는 이상치일까? 라고 고민이 
