# HW1 House price Prediction

## Relative
> Kaggle : https://www.kaggle.com/competitions/machine-learning-2023-nycu-regression

> Colab  : https://colab.research.google.com/?authuser=0#create=true


## 目標

給予資料(train_data.csv、test_data.csv、valid.csv)利用train、valid 資料裡面的feature去預測房價，並且在Kaggle做排名，下圖是最後跑出來的成果(結束後)。
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/927b289b-b1ff-40b6-bf46-e72f53c7eda4)
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/cb470d6b-93c2-4aa8-ad64-981d39f1238d)
### 初始想法
想說建造一個NN架構去跑，後來發現LOSS一直跑在10萬多降不下去(可能模型架構、學習率、BATCHSIZE等有關)後來就放棄了...
[`code`](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/HW1/House_price_prediction.ipynb)
### 第二次做
用了一些資料處理，並且使用隨機森林樹回歸(Random Forest Regression)去做，發現效果直接降到10萬以下
### 最後一次做
聽了跑分前三名的做法，都是用stacking的做法，資料前處理每個人都差不了多少(幾乎都使全部下去train幾乎都使全部下去train)，自己也使用看看，最後也真的跑了不錯的成績。
## Feature
| id  | 	a notation for a house  | 	Numeric  |
| ------------- |:-------------:|:-------------:|
| year  | 	date house was sold  | 	String  |
| month	  | date house was sold  | 	String  |
| day	  | date house was sold	  | String
price	  | Price is prediction target	  | Numeric
bedrooms  | 	Number of Bedrooms/House  | 	Numeric
bathrooms  | 	Number of bathrooms/bedrooms  | 	Numeric
sqft_living	  | square footage of the home  | 	Numeric
sqft_lot  | 	square footage of the lot  | 	Numeric
floors	  | Total floors (levels) in house	  | Numeric
waterfront  | 	House which has a view to a waterfront	  | Numeric
view  | 	Has been viewed  | 	Numeric
condition  | 	How good the condition is ( Overall )  | 	Numeric
grade  | 	overall grade given to the housing unit  | 	Numeric
sqft_above  | 	square footage of house apart from basement  | 	Numeric
sqft_basement  | 	square footage of the basement  | Numeric
yr_built  | 	Built Year  | 	Numeric
yr_renovated  | 	Year when house was renovated  | 	Numeric
zipcode  | 	zip	| Numeric
lat	  | Latitude coordinate  | 	Numeric
long  | 	Longitude coordinate  | 	Numeric
sqft_living15  | 	Living room area  | 	Numeric
sqft_lot15  | 	lotSize area  | 	Numeric
## Data Process
1. 方便處理與觀看，把train、valid data做合併，一起做處理
```
train_data=pd.concat([train_data,valid_data], axis=0,ignore_index=True) # 把index標籤弄掉
```
2. 對PRICE做LOG1P，把右斜曲線變回到正規畫曲線，如下圖
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/bbc7df5a-92bb-4c00-8306-c2c61e69d8e9)
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/be1b302d-8ce0-487a-90c2-2333b8adb720)

3. 找出Price與其他Feature的Corrrlation，但是最後沒用....(發現全部帶進去會比較好)
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/e6f849c3-7b38-45fb-91fb-78e9403e6798)

4. 把每個feature紀錄出來，看是否正規畫，如果沒有正規劃，就一樣對feature做log1p

``` python=
for feature in data.columns:
  data[feature] = np.log1p(data[feature])
  if data[feature].isnull().sum()>0:
    print(f"Have Nan in {feature}")
    data=data.drop(labels=feature,axis=1)
```

## Stacking
這次利用了五個模型(第一層)去做分別是 AdaBoostRegressor、RandomForestRegressor、GradientBoostingRegressor、CatBoostRegressor、LGBMRegressor
並且設定一定的iteration次數去跑，也記錄每個model的Loss Dacay
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/28ecc0f0-4558-473a-a47e-d0e2852ceeb1)

### Method1
利用每個模型跑出的結果，根據每個模型的MSE，給予一定的權重(weight)，得到最終的結果
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/8094849e-3135-48af-bc24-664079c60602)
### Method2
使用XGBboost，使用第一層的模型的預測結果當作訓練資料，再去做一次訓練。不一定要用XGBboost，可以用其他的模型也沒關係，可以用LinearRegression或訓練自己的神經網路即可。
### Croos Validation(還未做)
將訓練資料進行分組，一部分做為訓練子集來訓練模型，另一部分做為驗證子集來評估模型。用訓練子集的數據先訓練模型，然後用驗證子集去跑一遍，看驗證集的損失函數(loss)或是分類準確率等。
#### **優點:**

* 降低模型訓練對於資料集的偏差。
* 訓練集與驗證集完整被充分利用與學習。
## Model Loss Decay
由於 colab跑每個模型，所以跑很久，所以跑不到500就放沒繼續跑了，Best performance 是跑1000次。



* AdaBoostRegressor
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/f807ec21-ea0e-4431-a3f1-360cc96d810a)

* RandomForestRegressor
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/24861496-cf43-44cd-b52c-fb17d5fbd758)

* GradientBoostingRegressor
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/84df608d-b1e4-44ed-9c04-972fa92ec9b9)

* CatBoostRegressor
![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/06e2ecef-123f-43dc-b07f-eef0a883c8f3)

* LGBMRegressor:
*  ![image](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/assets/144834549/1b0163ec-32a6-415e-944d-dae56c1fee5e)


## 改善部分
* 可以使用更多model去做，由於每個model學習到的feature都不一樣，所以綜合所有的model能力會遠大於一個，可以增加許多model(SVM、Linear、NN)，也有一句俗諺：「三個臭皮匠，勝過一個諸葛亮。」
* 資料的前處理，對於正規劃可以使用其他的函數，如下:
    * MinMaxScaler：用于进行最小-最大缩放。
    * StandardScaler：用于进行标准化。
* 有給經緯度，可以去查美國那邊的房價登入，說不定有些地區特別貴，再去做分類也可以
* 因為有Feature裡面有時間，房價又隨著時間在增加，而且每年的房價，會有幾個月是最低的、最高的(可能呈現週期性)，可以利用這點去做分類
