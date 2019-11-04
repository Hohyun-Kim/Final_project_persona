# Latent Dirichlet Allocation (LDA)
### 출처: ratsgo 블로그

## LDA(Latent Dirichlet Allocation)
- Topic Modeling 중의 하나
- 주어진 문서에 대하여 각 문서에 어떤 주제들이 존재하는지에 대한 확률모형
- 토픽별 단어의 분포, 문서별 토픽의 분포를 모두 추정
- Bag-of-Words 가정 : 문서의 순서를 무시하고 빈도로 체크한다.
![title](http://i.imgur.com/r5e5qvs.png)
- 위 'Topic proportions & assignments'가 LDA 핵심 프로세스

```python
import random
from collections import Counter

random.seed(0)

# topic 수 지정
K=4

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
             
             
# 각 토픽이 각 문서에 할당되는 횟수
# Counter로 구성된 리스트
# 각 Counter는 각 문서를 의미
document_topic_counts = [Counter() for _ in documents]

# 각 단어가 각 토픽에 할당되는 횟수
# Counter로 구성된 리스트
# 각 Counter는 각 토픽을 의미
topic_word_counts = [Counter() for _ in range(K)]

# 각 토픽에 할당되는 총 단어 수
# 숫자로 구성된 리스트
# 각각의 숫자는 각 토픽을 의마함
topic_counts = [0 for _ in range(K)]

# 각 문서에 포함되는 총 단어 수
# 숫자로 구성된 리스트
# 각각의 숫자는 각 문서를 의미함
document_lengths = list(map(len, documents))

# 단어 종류의 수
distinct_words = set(word for document in documents for word in document)
V = len(distinct_words)

# 총 문서의 수
D = len(documents)

[Input 1]     [[random.randrange(K) for word in document] for document in documents]
```

```
[Output 1]    [[3, 3, 0, 2, 3, 3, 2],
               [3, 2, 1, 1, 2],
               [1, 0, 2, 1, 2, 0],
               [0, 2, 3, 0, 2],
               [3, 2, 1, 3],
               [3, 2, 0, 0, 0, 3],
               [0, 3, 2, 1],
               [2, 0, 1, 1],
               [1, 1, 3, 0],
               [0, 2, 3, 0],
               [2, 2, 0],
               [2, 1, 2, 3],
               [0, 3, 2],
               [1, 2, 1, 1, 1],
               [0, 2, 3]]
```

```python

def p_topic_given_document(topic, d, alpha=0.1):
    # 문서 d의 모든 단어 가운데 topic에 속하는
    # 단어의 비율 (alpha를 더해 smoothing)
    numerator = document_topic_counts[d][topic] + alpha
    denominator = document_lengths[d] + K * alpha
    A = numerator / denominator
    return A

def p_word_given_topic(word, topic, beta=0.1):
    # topic에 속한 단어 가운데 word의 비율
    # (beta를 더해 smoothing)
    numerator = topic_word_counts[topic][word] + beta
    denominator = topic_counts[topic] + V * beta
    B = numerator / denominator
    return B

def topic_weight(d, word, k):
    # 문서와 문서의 단어가 주어지면
    # k번째 토픽의 weight를 반환
    A = p_word_given_topic(word, k)
    B = p_topic_given_document(k, d)
    return A * B
    
    
def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k) for k in range(K)])

def sample_from(weights):
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0:
            return i
            

# 각 단어를 임의의 토픽에 랜덤 배정
document_topics = [[random.randrange(K) for word in document] for document in documents]

# 위와 같이 랜덤 초기화한 상태에서
# AB를 구하는 데 필요한 숫자를 세어봄
for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1
        
[Input 2]     document_topic_counts
```

```
[Output 2]    [Counter({0: 4, 1: 2, 3: 1}),
               Counter({2: 2, 1: 2, 3: 1}),
               Counter({3: 2, 2: 2, 0: 2}),
               Counter({3: 1, 2: 1, 1: 2, 0: 1}),
               Counter({2: 2, 0: 1, 1: 1}),
               Counter({1: 2, 2: 1, 3: 1, 0: 2}),
               Counter({1: 1, 0: 3}),
               Counter({0: 2, 1: 1, 3: 1}),
               Counter({0: 3, 2: 1}),
               Counter({0: 2, 1: 2}),
               Counter({3: 1, 1: 1, 0: 1}),
               Counter({0: 2, 3: 1, 2: 1}),
               Counter({0: 2, 1: 1}),
               Counter({2: 2, 3: 1, 1: 1, 0: 1}),
               Counter({3: 1, 0: 2})]
```

```python
[Input 3]     topic_word_counts
```

```
[Output 3]    [Counter({'Hadoop': 2,
                        'Big Data': 3,
                        'Spark': 1,
                        'Storm': 1,
                        'numpy': 1,
                        'pandas': 2,
                        'probability': 2,
                        'regression': 2,
                        'C++': 2,
                        'Haskell': 1,
                        'mathematics': 1,
                        'theory': 1,
                        'machine learning': 1,
                        'Mahout': 1,
                        'neural networks': 1,
                        'artificial intelligence': 2,
                        'statsmodels': 1,
                        'Python': 1,
                        'MongoDB': 1,
                        'support vector machines': 1}),
               Counter({'HBase': 1,
                        'Java': 2,
                        'MongoDB': 1,
                        'Cassandra': 1,
                        'statistics': 2,
                        'regression': 1,
                        'decision trees': 1,
                        'Python': 1,
                        'programming languages': 1,
                        'scikit-learn': 1,
                        'MapReduce': 1,
                        'R': 2,
                        'MySQL': 1}),
               Counter({'NoSQL': 1,
                        'Postgres': 1,
                        'scipy': 1,
                        'statsmodels': 1,
                        'Python': 1,
                        'machine learning': 1,
                        'libsvm': 1,
                        'R': 1,
                        'deep learning': 1,
                        'probability': 1,
                        'databases': 1,
                        'HBase': 1}),
               Counter({'Cassandra': 1,
                        'HBase': 1,
                        'Python': 1,
                        'scikit-learn': 1,
                        'R': 1,
                        'Java': 1,
                        'neural networks': 1,
                        'statistics': 1,
                        'deep learning': 1,
                        'Postgres': 1,
                        'libsvm': 1})]
```

```python
for _ in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):
            # 깁스 샘플링 수행을 위해
            # 샘플링 대상 word와 topic을 제외하고 세어봄
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1
            
            # 깁스 샘플링 대상 word와 topic을 제외한
            # 말뭉치 모든 word의 topic 정보를 토대로
            # 샘플링 대상 word의 새로운 topic을 선택
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
            
            # 샘플링 대상 word의 새로운 topic을 반영해
            # 말뭉치 정보 업데이트
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

[Input 4]     document_topics
```

```
[Output]      [[2, 2, 2, 2, 2, 2, 2],
               [1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 3, 3],
               [0, 3, 1, 3],
               [0, 0, 2, 0, 0, 0],
               [3, 3, 1, 3],
               [0, 0, 0, 3],
               [2, 2, 2, 2],
               [2, 2, 2, 2],
               [0, 0, 0],
               [2, 2, 2, 3],
               [0, 0, 0],
               [1, 1, 1, 1, 1],
               [2, 3, 1]]
```

```python
[Input 5]     document_topic_counts[0]
```

```
[Output 5]    Counter({0: 0, 1: 0, 3: 0, 2: 7})
```

```python
[Input 6]     topic_word_counts[0].most_common()
```

```
[Output 6]    [('Python', 4),
               ('R', 4),
               ('pandas', 2),
               ('machine learning', 2),
               ('statsmodels', 2),
               ('scikit-learn', 2),
               ('statistics', 2),
               ('numpy', 1),
               ('C++', 1),
               ('Haskell', 1),
               ('Mahout', 1),
               ('scipy', 1),
               ('programming languages', 1),
               ('Hadoop', 0),
               ('Big Data', 0),
               ('Spark', 0),
               ('Storm', 0),
               ('probability', 0),
               ('regression', 0),
               ('mathematics', 0),
               ('theory', 0),
               ('neural networks', 0),
               ('artificial intelligence', 0),
               ('MongoDB', 0),
               ('support vector machines', 0),
               ('decision trees', 0),
               ('MapReduce', 0),
               ('libsvm', 0),
               ('deep learning', 0),
               ('Java', 0),
               ('NoSQL', 0),
               ('HBase', 0),
               ('Cassandra', 0),
               ('databases', 0),
               ('Postgres', 0),
               ('MySQL', 0)]
```
