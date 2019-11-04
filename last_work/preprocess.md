# Preprocess

```python
# 기본 라이브러리 호출
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 한국어 전처리를 위한 라이브러리 호출
from konlpy.tag import Komoran, Okt

# Garbage Collect 호출
import gc

# 한국어 자연어처리를 위한 라이브러리 호출
import sys
sejong_corpus_cleaner_dir = "C:/research_persona/sejong_corpus_cleaner"
sys.path.append(sejong_corpus_cleaner_dir)

import sejong_corpus_cleaner

soylemmma_dir = "C:/research_persona/korean_lemmatizer"
sys.path.append(soylemmma_dir)

from soylemma import Lemmatizer
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

# 분석에 사용할 네이버 영화 평점 리뷰 데이터 로드
train_data = pd.read_csv('../nsmc/ratings_train.txt', sep='\t')
test_data = pd.read_csv('../nsmc/ratings_test.txt', sep='\t')

# Null값 제거
train_data = train_data.dropna(how='any')
test_data = test_data.dropna(how='any')
```

```python
train_data.head(10)

        id	document            label
0	9976970	아 더빙.. 진짜 짜증나네요 목소리 0
1	3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나 1
2	10265843	너무재밓었다그래서보는것을추천한다 0
3	9045019	교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정 0
4	6483659	사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...	1
5	5403919	막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.	0
6	7797314	원작의 긴장감을 제대로 살려내지못했다.	0
7	9443947	별 반개도 아깝다 욕나온다 이응경 길용우 연기생활이몇년인지..정말 발로해도 그것보단...	0
8	7156791	액션이 없는데도 재미 있는 몇안되는 영화	1
9	5912145	왜케 평점이 낮은건데? 꽤 볼만한데.. 헐리우드식 화려함에만 너무 길들여져 있나?	1
```

```python
train_data['document'].iloc[:5]

0                                  아 더빙.. 진짜 짜증나네요 목소리
1                    흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
2                                    너무재밓었다그래서보는것을추천한다
3                        교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정
4    사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...
Name: document, dtype: object
```

## 띄어쓰기가 수행되지 않은 텍스트 발견, 이를 처리하기 위한 시도들

```python
okt = Okt()
okt.morphs('너무재밓었다그래서보는것을추천한다', stem=True)

['너', '무재', '밓었', '다그', '래서', '보다', '추천', '한', '다']
```

### 위의 형태소 분석이 제대로 수행됬다고 볼 수 있을까?
- 내가 한다면 어떻게 수행할까?
- ['너무', '재밓었다', '그래서', 보다', '추천한다']
- 위와 같은 식으로 분리되어야 제대로 전처리됬다고 볼 수 있지 않을까?
- 그렇다면 어떻게 할 수 있을까?

```python
# 아래와 같이 score를 알 수 있다고 가정해보자.
tokenizer = MaxScoreTokenizer(scores={'너무':.7, '그래서':.7, '추천한다':.7})
tokenizer.tokenize('너무재밓었다그래서보는것을추천한다')
```

- 위에 대한 score를 어떻게 책정할 수 있을까?
- 이미 알려진 단어 사전이 존재한다면?
- 세종 말뭉치를 활용해보면 어떨까?

## 세종 말뭉치 데이터를 활용한 전처리 적용
- 20190906

```python
# 기본 라이브러리 호출
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 자연어처리를 위한 라이브러리 호출
from konlpy.tag import Komoran, Okt
import pickle
import os
from glob import glob

df = pd.read_csv('raw_data_nsmc.csv')

import pickle
with open('sejong_corpus_li.pkl', 'rb') as f:
    sejong_corpus_li = pickle.load(f)
    
# 한국어 전처리를 위한 라이브러리 호출
from konlpy.tag import Komoran, Okt

# Garbage Collect 호출
import gc

# 한국어 자연어처리를 위한 라이브러리 호출
import sys
sejong_corpus_cleaner_dir = "C:/research_persona/sejong_corpus_cleaner"
sys.path.append(sejong_corpus_cleaner_dir)

import sejong_corpus_cleaner

soylemmma_dir = "C:/research_persona/korean_lemmatizer"
sys.path.append(soylemmma_dir)

from soylemma import Lemmatizer
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

# 모든 단어에 대한 weight를 똑같이 주자.
score1 = {i[0] : .7 for i in sejong_corpus_li}

# 생각대로 잘 수행하는 것을 확인할 수 있다.
tokenizer = MaxScoreTokenizer(scores=score1)
tokenizer.tokenize('너무재밓었다그래서보는것을추천한다')

['너무', '재밓었다', '그래서', '보는것을', '추천', '한다']

# 이런 것을 잘 못한다...
tokenizer = MaxScoreTokenizer(scores=score1)
tokenizer.tokenize('과연이것도너가분리할수있을까후아하하하하키키키')

['과연', '이것', '도너가분리할수있을까', '후아하하하하키키키']
```

### 전처리에서 중요한 부분은?
- 의미를 파악할 수 있게 형태소 단위로 분리
- 어간으로 통일, 오탈자 전처리

### 위의 문장에 대한 형태소 분석이 실패한 이유는 뭘까?
- 품사별 weight가 똑같다
- '이것도', '너가', '분리', 할수있을까' 등과 같이 무조건 품사별로 분리하는 것이 아닌 '의미를 지니는 최소 단위'로 분리해줄 필요가 있다.

## 어절 말뭉치를 받아와 다시 시도해보자

```python
with open('eojeol_corpus_li.pkl', 'rb') as f:
    eojeol_corpus_li = pickle.load(f)

corpus_li = eojeol_corpus_li + sejong_corpus_li
len(corpus_li) # 1335021

# 어절에 더 높은 점수를 부여
score2 = {i[0] : .7 if i[1] == 'Eojeol' else .5 for i in corpus_li}

tokenizer = MaxScoreTokenizer(scores=score2)
print(tokenizer.tokenize('너무재밓었다그래서보는것을추천한다'))
print(tokenizer.tokenize('과연이것도너가분리할수있을까후아하하하하키키키')) # 위에 보단 낫지만 다시 위에 것을 분석을 못한다..

['너무', '재밓', '었다', '그래', '서보는', '것을', '추천한다']
['과연', '이것도', '너가', '분리할', '수있을', '까후', '아하하하', '하키', '키키']

# 어절을 합치고 전부 같은 점수를 부여
score3 = {i[0] : .7 if i[1] == 'Eojeol' else .7 for i in corpus_li}

tokenizer = MaxScoreTokenizer(scores=score3)
print(tokenizer.tokenize('너무재밓었다그래서보는것을추천한다'))
print(tokenizer.tokenize('과연이것도너가분리할수있을까후아하하하하키키키')) # 아까보다 훨씬 괜찮은 결과를 보인다.

['너무', '재밓', '었다', '그래서', '보는', '것을', '추천한다']
['과연', '이것도', '너가', '분리할', '수있을', '까후', '아하하하', '하키', '키키']

from konlpy.tag import Okt, Hannanum, Kkma, Mecab, Komoran

kom = Komoran()
okt = Okt()
han = Hannanum()
# mec = Mecab()
kkma = Kkma()

print(okt.morphs('과연이것도너가분리할수있을까후아하하하하키키키'))
print(han.morphs('과연이것도너가분리할수있을까후아하하하하키키키'))
print(kom.morphs('과연이것도너가분리할수있을까후아하하하하키키키'))
print(kkma.morphs('과연이것도너가분리할수있을까후아하하하하키키키'))

['과연', '이', '것', '도', '너', '가', '분리', '할수있을까', '후', '아하하하하', '키키', '키']
['과연이것도너가분리할수있을까후아하하하하키키키']
['과연', '이', 'ㄹ', '것', '도', '너', '가', '분리', '하', 'ㄹ', '수', '있', '을까', '후', '아하', '하하', '하키', '키키']
['과연', '이것', '도', '너', '가', '분리', '하', 'ㄹ', '수', '있', '을까', '후', '아하하', '하하', '키', '키', '키']

s = '너무재밓었다그래서보는것을추천한다'
print(okt.morphs(s))
print(han.morphs(s))
print(kom.morphs(s))
print(kkma.morphs(s))

['너', '무재', '밓었', '다그', '래서', '보는것을', '추천', '한', '다']
['너무재밓었다그래서보는것을추천한다']
['너무재밓었다그래서보는것을추천한다']
['너무', '재', '밓', '어', '었', '다', '그래서', '보', '는', '것', '을', '추천', '하', 'ㄴ다']

def gen_score(corpus_li, weights):
    score = {}
    for text, pos in corpus_li:
        if pos == 'Eojeol':
            score[text] = weights['Eojeol']
        elif pos == 'Adjective':
            score[text] = weights['Adjective']
        elif pos == 'Adverb':
            score[text] = weights['Adverb']
        elif pos == 'Determiner':
            score[text] = weights['Determiner']
        elif pos == 'Eomi':
            score[text] = weights['Eomi']
        elif pos == 'Exclamation':
            score[text] = weights['Exclamation']
        elif pos == 'Josa':
            score[text] = weights['Josa']
        elif pos == 'Noun':
            score[text] = weights['Noun']
        elif pos == 'Number':
            score[text] = weights['Number']
        elif pos == 'Pronoun':
            score[text] = weights['Pronoun']
        elif pos == 'Symbol':
            score[text] = weights['Symbol']
        elif pos == 'Verb':
            score[text] = weights['Verb']
        else:
            score[text] = 0
    return score

pos = ['Adjective', 'Adverb', 'Determiner', 'Eojeol', 'Eomi', 'Exclamation', 'Josa', 'Noun', 'Number', 'Pronoun', 'Symbol', 'Verb']
# wgt = [         .0,       .0,           .0,       .0,     .7,            .7,     .7,     .7,       .7,        .7,       .7,     .7]
wgt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
wgt[0] = .9
wgt[1] = .9
wgt[2] = .9
wgt[3] = .3
wgt[4] = .9
wgt[5] = .9
wgt[6] = .9
wgt[7] = .7

wgt[9] = .9
score = gen_score(corpus_li, dict(zip(pos, wgt)))

tokenizer = MaxScoreTokenizer(scores=score)
print(tokenizer.tokenize('너무재밓었다그래서보는것을추천한다'))
print(tokenizer.tokenize('과연이것도너가분리할수있을까후아하하하하키키키')) # 아까보다 훨씬 괜찮은 결과를 보인다.

['너무', '재밓', '었다', '그래서', '보는', '것을', '추천', '한다']
['과연', '이것', '도너', '가분', '리', '할수', '있', '을까', '후', '아하하하', '하키', '키키']
```

#### 분리 vs 가분

```python
# 어절을 합치고 전부 같은 점수를 부여
score3 = {i[0] : .7 if i[1] == 'Eojeol' else .7 for i in corpus_li}

tokenizer = MaxScoreTokenizer(scores=score3)
print(tokenizer.tokenize('너무재밓었다그래서보는것을추천한다'))
print(tokenizer.tokenize('과연이것도너가분리할수있을까후아하하하하키키키')) # 아까보다 훨씬 괜찮은 결과를 보인다.
print(tokenizer.tokenize('지루하지는 않은데 완전 막장임... 돈주고 보기에는....'))

['너무', '재밓', '었다', '그래서', '보는', '것을', '추천한다']
['과연', '이것도', '너가', '분리할', '수있을', '까후', '아하하하', '하키', '키키']
['지루하지', '는', '않은데', '완전', '막장', '임...', '돈주고', '보기에는', '....']
```

## 생각만큼 전처리가 잘 되지 않는다... 다른 툴은 없을까?
- soyspacing
- 핑퐁 띄어쓰기 
