# BA_03 Anomaly Detection(이상치 탐지)

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/201814625-adf72ae0-4f94-4550-98f1-4986626f82b6.PNG">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Anomaly Detection(이상치 탐지)을 바탕으로 만들어졌습니다.
이번에는 이상치탐지방식의 기본적인 개념과 기법중에서도 Model-based learning에서 Auto encoder에 대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.
## 목차
### 1. [BA_03 Anomaly Detection(이상치 탐지)_개념설명](#ba_03-anomaly-detection이상치-탐지_개념설명)
### 2. [BA_03 Anomaly Detection(이상치 탐지)_Model_based_learning Auto encoder](#ba_03-anomaly-detection이상치-탐지_model_based_learning-auto-encoder)
### 3. [BA_03 Anomaly Detection(이상치 탐지)_Model_based_learning Auto encoder 실습코드](ba_03-anomaly-detection이상치-탐지_model_based_learning-auto-encoder)
  #### 3.1. [데이터 설명](데이터_설명)
  #### 3.2. [코드](코드)
  #### 3.3. [결론](결론)

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

## BA_03 Anomaly Detection(이상치 탐지)_Model_based_learning Auto encoder
  
이상치 탐지가 어떤것인지 아셨나요? 그럼 이번에는 모델을 통한 이상치탐지기법중에 Auto encoder에 대해 알아보려고합니다. 
  
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

데이터들의 feature는 Time(시간), V1~V28(개인정보로 인해 공개되어지지 않은값임 PCA로 변환된 값), Amount(거래 금액), Class(사기여부로 1이면 사기를 당했고 0이면 정상임)로 구성되었으며 Null값은 없는 데이터이다.

- ### 코드 

```python
#필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

%matplotlib inline
# 시각화 라이브러리 설정
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
```
필요한 모듈 및 라이브러리를 불러오고 그래프의 스타일이니 색깔, 사이즈등을 미리 설정한다.

```python
#데이터를 불러오고 df.head()를 통해 데이터의 형태를 확인해봄
df = pd.read_csv("creditcard.csv")
df.head()
```
<p align="center"><img width="680" alt="image" src="https://user-images.githubusercontent.com/97882448/202340245-bfe970b3-56e3-4569-8e79-0b5712cb1ac6.png">

```python
#df.shape를통해 데이터의 형태를 확인함
df.shape
#데이터에 Null값이 있는지 확인해봄
df.isnull().values.any()
# RANDOM_SEED와 Class의 LABELS설정
RANDOM_SEED = 2022
LABELS = ["Normal", "Fraud"]
```
데이터의 형태 및 Null값이 있는지 확인하고 Random_SEED를 설정해줌 출력값은 값이 첨부한 Jupyter_Notebook을 통해 확인가능함

```python
count_classes = pd.value_counts(df['Class'])
#rot은 글자를 회전 시킴
count_classes.plot(kind = 'bar', rot=45, color="lightskyblue")
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/202346218-c8e15721-f581-477c-82bd-ae15b3381dc1.png">

bar plot으로 그려보려고하지만 Fraud의 갯수가 너무 작아서 bar plot은 좋은그림이 아닌것 같음

```python
fraud = df[df.Class == 1]
normal = df[df.Class == 0]
fraud.shape #(492, 31)
normal.shape #(284315, 31)

#Class의 갯수를 세수 표로 만듦 또한 reset_index()를 통하여 index에 대한 표도 만들어줌 
table = df['Class'].value_counts().to_frame().reset_index()
# 전체의 데이터에서 몇 프로를 차지 하는지 표를 추가하고 소수점 4째자리까지 추출
table['Percent(%)'] = df["Class"].apply(lambda x : round(100*float(x) / len(data), 4))
#index와 Class의 이름을 바꾸어줌
table= table.rename(columns = {"index" : "Target", "Class" : "Count"})

table
```
갯수를 보아하니 왜 barplot에 fraud가 잘 나오지 않았는지 알것 같음 따라서 table을 생성해보니 아래의 그림처럼 사기거래의 비율이 0.1727% 였던것을 알수 있음
  
<p align="center"><img width="250" alt="image" src="https://user-images.githubusercontent.com/97882448/202347439-16aeef42-af79-454c-98a4-3e6bcedd830f.png">

```python
#사기거래 금액의 분포
frauds.Amount.describe()
```
<p align="center"><img width="251" alt="image" src="https://user-images.githubusercontent.com/97882448/202348405-5e9b0900-092c-4c43-a648-985762d9f58d.png">
  
```python
#정상거래 금액의 분포
normal.Amount.describe()
```
<p align="center"><img width="262" alt="image" src="https://user-images.githubusercontent.com/97882448/202348460-6888732c-55f7-4fea-9f24-2d57d3951ee2.png">

```python
#정상과 사기거래의 거래량과 금액을 graph로 나타냄
bins = np.linspace(200, 2500, 200)
plt.hist(normal.Amount, bins=bins, alpha=1, density=True, label='Normal')
plt.hist(frauds.Amount, bins=bins, alpha=1, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Transaction amount and Percentage of transactions")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions");
plt.show()
```
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/97882448/202349441-b84e1e14-4cd5-4703-a606-2e3fde8b4103.png">
  
정상과 사기거래의 거래량과 금액을 하나의 그래프로 겹쳐서 그려봄 x축을 거래금액으로 설정하고 y축을 거래횟수의 %인데 확실히 차이가 난다고 볼수있음

```python 
#정상거래와 사기거래의 시간(초)에 따른 거래량을 알아보고자함
#sub plot이 2개의 graph가 그릴수있도록 설정해주고  x축을 설정함
function,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
function.suptitle('Time of transaction and Amount by class')
#시간에 따른 사기거래의 거래량
ax1.scatter(frauds.Time, frauds.Amount,color="red")
ax1.set_title('Fraud Class')
#시간에 따른 정상거래의 거래량
ax2.scatter(normal.Time, normal.Amount,color="skyblue")
ax2.set_title('Normal Class')

plt.xlabel('Time_(Sec)')
plt.ylabel('Amount')
plt.show()
```
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/97882448/202351958-185b9e8e-b26d-443e-b8d2-866b7dcac6ae.png">
  
정상거래와 사기거래의 시간(초)에 따른 거래량을 알아보려고 x축을 공유하여 사기와 정상거래가 어떤 부분이 다른지 알아보고자하였으나 결과적으로, 정상거래나 사기거래나 시간에 비례해서 좋아보이진 않아보임

```python 
from sklearn.preprocessing import StandardScaler
#시간이 그렇게 중요한 요소가 아니라 판단되어 시간을 지움 
data = df.drop(['Time'], axis=1)
#거래량을 StandardScaler를 통해 값들을 스케일링 함, 이유는 평균을 제거하고 데이터를 단위 분산으로 조정하기에 이상치가 있다면 데이터의 확산은 매우 달라져서 이상치에 매우 민감
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
```
먼저, 시간이 이상치탐지에 그렇게 중요한 요소가 아니라고 생각되어 시간열을  drop하고 amount인 거래량을 StandardScaler를 통해 스케일링하여 데이터가 이상치에 매우 민감하게 반응하도록 만듦
  
```python  
X_train, X_test = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
#이상치탐지에는 중요한특성이 있는데 train데이터는 정상인 데이터만 사용함
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

X_train.shape
```
이제 데이터들을 훈련시켜야 함 이상치탐지할때 중요한요소가 있는데 그것중 하나는 **train데이터는 정상 데이터**만 사용해야함
test_size를 0.3으로 설정하고 random_state는 아까 설정한 2022로 설정됨 그러면 X_train.shape은 (199000, 29)로 설정됨
  
```python   
#X_train의 열의 갯수:28개
input_dim = X_train.shape[1]

encoding_dim = 14

input_layer = Input(shape=(input_dim, ))
#regularizers를 L1으로 설정 하였음
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()
```
<p align="center"><img width="578" alt="image" src="https://user-images.githubusercontent.com/97882448/202365189-72e2aa85-c61a-4cc4-a925-ba496ef91c17.png">

위에 Autoencoder는 4개의 fully connected layer로 만들어져 있으며, 각 layer는 14, 7, 7, 29개로 구성됨 
모델을 요약하면 위와 같은결과가 나옴

```python   
nb_epoch = 50
batch_size = 32
#오토인코더 컴파일함
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
#ModelCheckpoint로 성능 좋은모델을 저장함
checkpointer = ModelCheckpoint(filepath="model.h",
                               verbose=0,
                               save_best_only=True)
#TensorBoard는 TensorFlow에서 발생한 로그를 표시함
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
```
<p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/202366748-3887551d-32a6-432b-93d2-ab2a38b01111.png">

```python 
#데이터 로드
autoencoder = load_model('model.h')
#epoch에 따른 train과 test의 loss함수
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
```
<p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/202391533-12fe581d-4e52-4c08-af2b-2a749a371b05.png">
  
데이터를 로드 한다음에 위 표를 보면 epoch에 따른 train과 test의 loss함수임
  
```python 
predictions = autoencoder.predict(X_test)
#np.power(a,b)는 제곱연산을 할때 사용되면 a^b를 뜻함
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
percent=[0.001,0.1,0.9,0.999]
error_df.describe(percentiles=percent)
```
<p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/202398829-6b707f5d-215d-40ea-be0f-2d14da32ec65.png">

대부분의 데이터가 정상 데이터이기에 reconstruction_error의 평균도 0.736549밖에 되지 않으나 99.9%는 44.121528로 사기거래라고 나옴
 
```python 
# 정상데이터의 reconstruction_error분포
fig = plt.figure()
#111은 1행째의 1열의 첫 번째라는 의미임
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 20)]
normal_error_df= ax.hist(normal_error_df.reconstruction_error.values, bins=30, color = "skyblue")
```
<p align="center"><img width="450" alt="image" src="https://user-images.githubusercontent.com/97882448/202396791-24e43a2f-a2e4-4eae-acf4-37113207e563.png">
  
정상데이터의 reconstruction_error분포는 위 그림과 같음
 
```python 
# 사기데이터의 reconstruction_error분포
fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
fraud_error_df = ax.hist(fraud_error_df.reconstruction_error.values, bins=30 ,color = "red")
```  
<p align="center"><img width="450" alt="image" src="https://user-images.githubusercontent.com/97882448/202399705-a624b26e-07a9-494e-a197-cefc9bd3d591.png">
  
사기데이터의 reconstruction_error분포는 위 그림과 같음
 
```python 
#sklearn의 기법들 
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('ROC')
#AUROC를 계산
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.ylabel('True Positive Rate(TPR)')
plt.xlabel('False Positive Rate(FPR)')
plt.show();

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'pink', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()
  
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()
```
<p align="center"><img width="1000" alt="image" src="https://user-images.githubusercontent.com/97882448/202400810-310a38e9-1724-4c93-8577-215ed295cab4.png">

위 그림을 보면 Reconstruction error가 증가할수록 정밀도가 올라가는 것을 알수 있음

```python 
threshold = 3
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
```
![image](https://user-images.githubusercontent.com/97882448/202403830-cd68a650-5c4a-42b5-a96c-7cef31998fa3.png)

```python 
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/202404021-162db7c1-56aa-494a-9f5c-58eb9ed4cef0.png">

threshold=3에 따른 confusion matrix임 실제 사기거래였을때 정상이라고 하는 상황을 최대한 줄이기 위해서 threshold를 생각보다 낮게 잡았음

- ### 결론
  
여기서 사용한 데이터셋은 imblance 하기에 높은 정확도(accuracy)만 보면 좋은 모델이라고 착각할수 있지만 낮은 재현률(recall)과 정밀도(precision)를 보임
recall과 precision을 개선하기 위한 것으로는 다른 이상치탐지방법이나 AE의 구조를 바꿔보면 좋을것 같음 
  
---
 ### Reference
 1. https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 <강필성교수님 자료>
 2. https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd <Credit Card Fraud Detection using Autoencoders in Keras >
