# WIL 7
이번엔 좀 짧다~~(왜 시험이 3개?왜 시험이 3개?왜 시험이 3개?)~~
# 비지도학습
비지도학습은 독립변수와 종속변수간의 관계나 상관을 학습하는 모형이라면, 비지도학습은 독립변수들만을 가지고 그 특징을 학습하는 모델이다. 크게 군집하아 차원축소가 있다
# 군집화(Clustering):
군집화는 데이터들에 대해서 임의의 기준을 통해 데이터들을 구분지어 군집(클러스터)로 묶는 것을 말한다. k-Means, DBSCAN 등의 알고리즘이 있다,
## k-Means
군집의 갯수를 미리 지정해야 하는 방법이다. k개의 군집 중심을 임의로 배치하고, 데이터들을 가장 가까운 중심에 종속시킨다. 그 후 속한 데이터들의 평균쪽으로 증심점을 이동시킨다. 그 후 중심점이 특정 지점으로 수렴할 때 까지 이를 계속 반복한다.
## DBSCAN
저번시간에 사용한 k-NN알고리즘과 유사하게 먼저 거리 e와 최소표본갯수 n을 정해야 한다.   
먼저 각 표본들에 대해 그 표본을 중심으로 유클리드 거리로 e 안에 다른 표본들이 n개 이상 존재하는지 확인한다. 존재한다면, 그 표본을 해당 군집의 핵심표본으로 정한다.   
만약 어떤 표본에 대해서 이 과정을 시행하였는데 스스로가 핵심표본이 되지는 못하지만 자신과의 거리 e 안에 표본중심이 존재한다면, 이 표본을 경계표본으로 정한다.
마지막으로 서로 포함하는 핵심표본과 경계표본들을 하나로 묶어 하나의 군집으로 정한다.
동심(同心) 형태와 같이 평균만을 사용하였을 때 군집상에 큰 차이를 발견하지 못하는 경우에 k-Means 알고리즘보다 가하학적으로 적절한 군집을 선택하는데에 있어 보다 적합하다.
## GMM
GMM(Gaussian Mixture Model)을 사용한다. GMM은 EM방식을 이용하여 가우스 분포 모수를 추정하는 방식이다. 이를 군집화에도 적용시켜 각 데이터가 가상의 군집에 속할 확률을 계산하여 종속시켜 보고, 이에서 점진적으로 모수를 조정하여 모수를 최적화시키면서 클러스터를 정한다.
# 차원축소
ML에서 모델을 학습시키는데에 있어 보통 정보는 많으면 많을수록 좋다. 하지만 위에서 본 경우와 같이 알고리즘이 거리를 통해 분석하는 기법을 사용하거나, 결과에 무작위적인 독립변수가 섞여있다면, 모든 특성을 다 고려하는 것은 오히려 성능을 약화시킬 수 있다. 이를 방지하기 위해 특성들의 수를 축소시키게 되는데, 이를 차원축소라고 한다.   Feature Selection과 Feature Extraction이 있다.
## Feature Selection
Feature Selection은 특성들을 일부만 고르는 것이다. 이를 달성하는 방법으로 크게 세가지가 있다:
### Filter 방법
특성들을 피어슨 상관분석, 카이제곱검정등을 통해 상관관ㅖ를 미리 파악한 뒤 특성을 골라내는 방식이다.
### Wrapper 방법
각 특성들의 경우의 수에 따라 실제로 학습을 해보고 정확도와 같이 적합하다고 생각하는 특성의 집합을 최종적으로 선택하는 방식이다.
### Embedded 방법
Wrapper방식에서 패널티 부과와 같이 추가적인 성능 검사를 추가적으로 시행한다.


보통 Filter방법보다 Wrapper, Embedded 방식이 더 비싸다.
## Feature Extraction
Feature Extraction은 특성들 중 일부를 선택하는 것이 아닌 여러 특성들을 하나의 특성으로 뭉치는 과정을 통해 총 특성의 수를 줄이는 기법이다.
### **PCA**
Feature Extraction 방법중 하나이며, xy축에서 벗어나 분산을 보존시키는 새로운 기저를 만들어 내는 기법이다.   
기본적으로 각 표본은 p개의 기저를 선형결합시킨 형태로 나타낼 수 있다. 이를 하나의 새로운 벡터로 만드는 과정으로 기저의 수를 줄여 특성의 수를 감소시키는 것이다. 
# 참고문헌
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html   
* https://ariz1623.tistory.com/224   
* https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py   
* https://bcho.tistory.com/1205   
* https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html   
* https://angeloyeo.github.io/2021/02/08/GMM_and_EM.html   
* https://roytravel.tistory.com/342   
* https://studying-haeung.tistory.com/14   
* https://zephyrus1111.tistory.com/420   
* https://zephyrus1111.tistory.com/183   
* https://bcho.tistory.com/1204   
* https://angeloyeo.github.io/2019/07/27/PCA.html   
* https://techblog-history-younghunjo1.tistory.com/106   
* https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/   
* https://m.blog.naver.com/euleekwon/221464171572   
* https://datascienceschool.net/03%20machine%20learning/18.*01%20%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88%20%ED%98%BC%ED%95%A9%EB%AA%A8%ED%98%95%EA%B3%BC%20EM%20%EB%B0%A9%EB%B2%95.html  
* https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/ 
