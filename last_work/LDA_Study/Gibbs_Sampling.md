# Gibbs Sampling
### 출처: ratsgo 블로그

```python
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def roll_a_dice():
    return random.choice(range(1, 7))

def direct_sample():
    d1 = roll_a_dice()
    d2 = roll_a_dice()
    return d1, d1+d2
    
def random_y_given_x(x):
    return x + roll_a_dice()

def random_x_given_y(y):
    if y <= 7:
        return random.randrange(1, y)
    else:
        return random.randrange(y-6, 7)

def gibbs_sample(num_iters=100):
    x, y = 1, 2
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

np.array([direct_sample() for _ in range(1000)])
```
```
array([[1, 2],
       [3, 6],
       [3, 7],
       ...,
       [5, 6],
       [1, 3],
       [3, 8]])
```
```python
ds = np.array([direct_sample() for _ in range(200)])
gs = np.array([gibbs_sample() for _ in range(200)])

fig = plt.figure(figsize=(20, 8), facecolor='w')

ax = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)

ax.scatter(ds[:, 0], ds[:, 1])
ax1.scatter(gs[:, 0], gs[:, 1])
```
![title](https://github.com/jinmang2/Final_project_persona/blob/master/last_work/LDA_Study/gibbssampling.png?raw=true)

## Dirichlet Distribution
- k차원의 실수 벡터 중 벡터의 요소가 양수이며 모든 요소를 더한 값이 1인 경우에 확률값이 정의되는 연속확률분포
