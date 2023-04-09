# WIL 4주차
이번에는 Kaggle 에서 `EDA To Prediction(DieTanic)`을 팔사하며 분석해보았다.   
유명(하고 오래된) EDA라고 한다. seaborn이 주로 사용되었다. seaborn의 버전이 옛날 꺼라 그대로 실행하면 오류가 좀 생기는 것 같다.   
~~(너무 길게 쓰니까 힘듦)~~
## matplotlib WIL
* plt.style.use()   
전반적인 디자인을 설정할 수 있다. 가능한 디자인은

```python
#import matplotlib.pyplot as plt 가 전제
plt.style.available
```

로 확인할 수 있다.   

* %matplotlib inline   
IPython에서 많이 이용되는데, 도표나 그림들을 바로 보여주는 코드라고 한다. notebook에서 결과를 바로 보여주는데에 사용되었다.
* matplotlib.axes.Axes.set_xticks, matplotlib.axes.Axes.set_yticks   
도표에서 각각 x축, y축의 눈금을 지정하는 메소드이다.   
## pandas
* pandas.DataFrame.value_counts()   
열의 각 값에 대한 횟수를 계산해 주는 메소드이다.
* pandas.core.groupby.DataFrameGroupBy.plot   
기본값으로 `matplotlib`를 사용하여 해당 `dataframe`에 대한 `plot`을 만들어 주는 메소드이다.   `.plot(kind="bar")` 이나 `.plot.bar()`과 같은 형태로 쓸 수 있다고 한다.
* pandas.Series.str.extract()   
For each subject string in the Series, extract groups from the first match of regular expression pat.
`Series`에 있는 각 `string`에서 해당하는 정규표현식을 만족하는 문자열 그룹을 반환하는 메소드이다.
* pandas.crosstab()   
한 변수를 기준으로 나머지 변수들의 개수나 수치를 파악하는데 사용되었던 메소드이다.
* pandas.DataFrame.T   
행혈에서의 전치행렬과 같이, 우상향-좌하향 대각선을 기준으로 뒤집는다.
## seaborn
* seaborn.countplot()
특정 변수를 기준으로 각 그룹마다 변수가 지는 깂의 개수를 세 표로 보여주는 메소드이다. pandas의 value_counts()를 한 변수에 대해 가능한 모든 경우의 수에 대해 해준다고 생각하면 될 것 같다.
* seaborn.catplot()
어떠한 범주, 또는 범위(나이대, 성별 등)를 나타내는 정보와 수치적인 정보 간의 관계를 보여주는 메소드이다. 이전엔 `factorplot`으로 사용되었으며, 스스로 `figure`을 만들기 때문에 figure을 따로 만들어 줄 필요가 없다. 
## EDA
* 지니고 있는 정보에 잘못 표기된 정보나 오타가 있을 수 있다. 이는 EDA 과정에서나 전처리 과정에서 오류나 이상치를 만들어 낼 수 있으므로 그 전에 꼭 확인하여야 한다.
* 정보가 None으로 존재하지 않더라도, 다른 정보들을 이용해서 합리적인 수치를 기입할 수 있다면 채워보는 것이 좋다. 이는 결측치를 제거하여 더 많은 데이터를 얻을 수 있다는 것이다.
## ETC
* 옛날 코드 쓸일 있으면 버전 체크랑 호환성 확인은 꼭 하자.




# 궁굼한 점
* 경칭을 regex로 추출해 낼때 `for`반복문을 사용한 이유가 궁금하다. 반복문이 없어도 정상적으로 작동되는 것으로 보이는데, 이것 또한 버전 문제인 걸까?
