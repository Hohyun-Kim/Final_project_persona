- 우도 계산하는 부분부터 막힘, 공부 후 추가 진행 요망
- Big O님 블로그로 공부

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

documents = [["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
             ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
             ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
             ["R", "Python", "statistics", "regression", "probability"],
             ["machine learning", "regression", "decision trees", "libsvm"],
             ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
             ["statistics", "probability", "mathematics", "theory"],
             ["machine learning", "scikit-learn", "Mahout", "neural networks"],
             ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
             ["Hadoop", "Java", "MapReduce", "Big Data"],
             ["statistics", "R", "statsmodels"],
             ["C++", "deep learning", "artificial intelligence", "probability"],
             ["pandas", "R", "Python"],
             ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
             ["libsvm", "regression", "support vector machines"]]

M = len(documents) # 문서의 수

Vocabrary = set([word for doc in documents for word in doc])

V = len(Vocabrary)
```

```python
plt.figure(figsize=(10, 8), facecolor='w')
plt.hist(np.random.dirichlet(np.arange(1, 1000001) / 1000001), bins=50); plt.show()
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig1.png?raw=true)

```python
plt.figure(figsize=(10, 8), facecolor='w')
plt.plot(np.random.dirichlet(np.arange(1, 1000001) / 1000001)); plt.show()
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig2.png?raw=true)

```python
plt.figure(figsize=(10, 8), facecolor='w')
plt.hist(np.random.beta(5, 5, 1000000), bins=20); plt.xlim([0, 1]); plt.show()
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig3.png?raw=true)

```python
np.random.dirichlet([1,1,1], 10000)
```
```
array([[0.15446638, 0.72389484, 0.12163878],
       [0.58855656, 0.16779795, 0.24364549],
       [0.3199086 , 0.44657544, 0.23351596],
       ...,
       [0.73314198, 0.16348383, 0.10337419],
       [0.55850089, 0.36290034, 0.07859877],
       [0.31383483, 0.14156726, 0.54459791]])
```
## 데이터 사이언스 스쿨
### 9.3 베이즈 추정법

```python
import seaborn as sns
sns.set_style('whitegrid')

# adjust 한글 font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

import scipy as sp

plt.figure(figsize=(10, 8), facecolor='w')
xx = np.linspace(0, 1, 1000)
a0, b0 = 1, 1
plt.plot(xx, sp.stats.beta(a0, b0).pdf(xx), c='r', ls='--', label='사전분포')
a1, b1 = a0 + 1, b0 + 3
plt.plot(xx, sp.stats.beta(a1, b1).pdf(xx), c='g', ls='-', label='사후분포')
# a1, b1 = a0 + 6, b0 + 4
# plt.plot(xx, sp.stats.beta(a1, b1).pdf(xx), c='g', ls='-', label='사후분포')
plt.legend(fontsize=12)
plt.title('베이즈 추정법으로 계산한 베르누이분포 모수의 분포', fontsize=12)
plt.show()
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig4.png?raw=true)

```python
plt.figure(figsize=(10, 8), facecolor='w')

mu0 = 0.65
a, b = 1, 1
print('초기 추정: 모드 = 모름')

xx = np.linspace(0, 1, 1000)
plt.plot(xx, sp.stats.beta(a, b).pdf(xx), ls=':', label='초기 추정')

np.random.seed(0)

for i in range(10):
    x = sp.stats.bernoulli(mu0).rvs(50)
    N0, N1 = np.bincount(x, minlength=2)
    a, b = a + N1, b + N0
    plt.plot(xx, sp.stats.beta(a, b).pdf(xx), ls='-.', label='{}차 추정'.format(i))
    print('{}차 추정: 모드 = {:4.2f}'.format(i, (a - 1) / (a + b - 2)))

plt.vlines(x=mu0, ymin=0, ymax=20)
plt.ylim([0, 20])
plt.legend(fontsize=12)
plt.title('베르누이분포의 모수를 베이즈 추정법으로 추정한 결과', fontsize=12)
plt.show()
```
```
초기 추정: 모드 = 모름
0차 추정: 모드 = 0.64
1차 추정: 모드 = 0.69
2차 추정: 모드 = 0.65
3차 추정: 모드 = 0.66
4차 추정: 모드 = 0.66
5차 추정: 모드 = 0.65
6차 추정: 모드 = 0.66
7차 추정: 모드 = 0.66
8차 추정: 모드 = 0.66
9차 추정: 모드 = 0.66
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig5.png?raw=true)

```python
import matplotlib.tri as mtri

def plot_dirichlet(alpha, n):
    
    n1 = np.array([1, 0, 0])
    n2 = np.array([0, 1, 0])
    n3 = np.array([0, 0, 1])
    n12 = (n1 + n2) / 2
    m1 = np.array([1, -1, 0])
    m2 = n3 - n12
    m1 = m1 / np.linalg.norm(m1)
    m2 = m2 / np.linalg.norm(m2)
    
    def project(x):
        return np.dstack([(x - n12).dot(m1), (x - n12).dot(m2)])[0]
    
    def project_reverse(x):
        return x[:, 0][:, np.newaxis] * m1 + x[:, 1][:, np.newaxis] * m2 + n12
    
    eps = np.finfo(float).eps * 10
    X = project([[1 - eps, 0, 0], [0, 1 - eps, 0], [0, 0, 1 - eps]])
    
    triang = mtri.Triangulation(X[:, 0], X[:, 1], [[0, 1, 2]])
    refiner = mtri.UniformTriRefiner(triang)
    triang2 = refiner.refine_triangulation(subdiv=6)
    XYZ = project_reverse(
        np.dstack([triang2.x, triang2.y, 1-triang2.x-triang2.y])[0])

    pdf = sp.stats.dirichlet(alpha).pdf(XYZ.T)
    plt.tricontourf(triang2, pdf, cmap=plt.cm.bone_r)
    plt.axis("equal")
    plt.title("정규분포 확률변수의 모수를 베이즈 추정법으로 추정한 결과: {} 추정".format(n))
    plt.show()

mu0 = np.array([0.3, 0.5, 0.2])
np.random.seed(0)

a0 = np.ones(3) / 3
plt.figure(figsize=(8, 6))
plot_dirichlet(a0, '초기')
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig6.png?raw=true)

```python
plt.figure(figsize=(8, 6))

x1 = np.random.choice(3, 50, p=mu0)
N1 = np.bincount(x1, minlength=3)
a1 = a0 + N1

print("종류별 붓꽃의 수 ={}".format(N1))
print("1차 추정 하이퍼모수:", (a1 - 1)/(a1.sum() - 3))

plot_dirichlet(a1, "1차")
```
```
종류별 붓꽃의 수 =[10 32  8]
1차 추정 하이퍼모수: [0.19444444 0.65277778 0.15277778]
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig7.png?raw=true)

```python
plt.figure(figsize=(8, 6))

x2 = np.random.choice(3, 50, p=mu0)
N2 = np.bincount(x2, minlength=3)
a2 = a1 + N2

print("종류별 붓꽃의 수 ={}".format(N2))
print("2차 추정 하이퍼모수:", (a2 - 1)/(a2.sum() - 3))

plot_dirichlet(a2, "2차")
```
```
종류별 붓꽃의 수 =[ 9 29 12]
2차 추정 하이퍼모수: [0.18707483 0.61564626 0.19727891]
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/image/fig8.png?raw=true)

## 우도 계산
