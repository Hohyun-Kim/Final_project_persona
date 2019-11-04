# 세종 말뭉치 데이터 정제
- 아래의 두 라이브러리 활용
    - [`sejong_corpus_cleaner`](https://github.com/lovit/sejong_corpus_cleaner)
    - [`sejong-corpus`](https://github.com/lovit/sejong_corpus)

```python
import sys
sejong_cleaner_git_repo = 'C:/research_persona/sejong_corpus_cleaner'
sys.path.append(sejong_cleaner_git_repo)

from sejong_corpus_cleaner.rawtext_loader import load_colloquial_text_as_eojeol_morphtags
from sejong_corpus_cleaner.rawtext_loader import load_written_text_as_eojeol_morphtags

path = '../sejong_corpus_cleaner/data/raw/'

%%time
from glob import glob
from sejong_corpus_cleaner.rawtext_loader import load_texts_as_eojeol_morphtags
# 세종 말뭉치의 원 데이터를 띄어쓰기로 구분된 '형태소/품사'열의 list of str로 변환
# 세종 말뭉치의 구어와 문어 데이터는 loading 함수가 다르기 때문에 데이터의 종류에 다라 
# is_cooloquial=True, False 옵션을 잘 부여해야 함

paths = list(
    map(
        lambda x : x.replace('\\', '/'), 
        glob('../sejong_corpus_cleaner/data/raw/colloquial/*.txt')
    )
)
eojeol_morphtag_colloquial = load_texts_as_eojeol_morphtags(paths, is_colloquial=True)

paths = list(
    map(
        lambda x : x.replace('\\', '/'), 
        glob('../sejong_corpus_cleaner/data/raw/written/*.txt')
    )
)
eojeol_morphtag_written = load_texts_as_eojeol_morphtags(paths, is_colloquial=False)

import pickle
eojeol_morphtag = (eojeol_morphtag_colloquial, eojeol_morphtag_written)
with open('eojeol_morphtag.pkl', 'wb') as f:
    pickle.dump(eojeol_morphtag, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## 말뭉치 작업
- 20190906
```python
import pickle
with open('eojeol_morphtag.pkl', 'rb') as f:
    eojeol_morphtag_colloquial, eojeol_morphtag_written = pickle.load(f)
    
import sys
sejong_cleaner_git_repo = 'C:/research_persona/sejong_corpus_cleaner'
sys.path.append(sejong_cleaner_git_repo)

from glob import glob

colloquial_paths = list(
    map(
        lambda x : x.replace('\\', '/'), 
        glob('../sejong_corpus_cleaner/data/raw/colloquial/*.txt')
    )
)

written_paths = list(
    map(
        lambda x : x.replace('\\', '/'), 
        glob('../sejong_corpus_cleaner/data/raw/written/*.txt')
    )
)

%%time
from sejong_corpus_cleaner.rawtext_loader import load_texts_as_eojeol_morphtags_table
# load_texts_as_eojeol_morphtags_table 함수는 세종 말뭉치의 원 데이터(raw data)로부터 어절을 구
# 구성하는 형태소와 해당 어절의 빈도수를 pd.DataFrame형태로 제공한다.

colloquial_table = load_texts_as_eojeol_morphtags_table(
    colloquial_paths, is_colloquial=True)
written_table = load_texts_as_eojeol_morphtags_table(
    written_paths, is_colloquial=False)
    
# Wall time: 3min 54s

from sejong_corpus_cleaner.simplifier._simplify import to_simple_tag_sentence0

colloquial_table['simplified_tag'] = colloquial_table['morphtags'].map(
    lambda x : to_simple_tag_sentence([tuple(i.split('/')) for i in x.split(' ')]))
    
colloquial_table['morphtags'] = colloquial_table['morphtags'].map(
    lambda x : [tuple(i.split('/')) for i in x.split(' ')])
    
written_table['morphtags'] = written_table['morphtags'].map(
    lambda x : [tuple(i.split('/')) for i in x.split(' ')])
    
err_li = []
for ix, i in enumerate(written_table['morphtags']):
    try:
        to_simple_tag_sentence(i)
    except:
        err_li.append(ix)
        
len(err_li) # 2210

written_err = written_table.loc[np.isin(np.arange(written_table.shape[0]), np.array(err_li))]
written_table = written_table.loc[~np.isin(np.arange(written_table.shape[0]), np.array(err_li))]

written_table.loc[:, 'morphtags'] = written_table['morphtags'].map(
    lambda x : [i for i in x if re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', i[0]) != '']).values
    
written_table = written_table[written_table['morphtags'].map(lambda x : len(x) > 0)]
written_table['simplified_tag'] = written_table['morphtags'].map(lambda x : to_simple_tag_sentence(x))

written_li = []
for ix, li in written_table['simplified_tag'].items():
    written_li.extend(li)
    
colloquial_li = []
for ix, li in colloquial_table['simplified_tag'].items():
    colloquial_li.extend(li)

len(written_li), len(colloquial_li) # (3844398, 385950)
len(set(written_li)), len(set(colloquial_li)) # (157862, 24553)
len(set(written_li).union(set(colloquial_li))) # 164485

sejong_corpus_li = list(set(written_li).union(set(colloquial_li)))
sejong_corpus_li = [i for i in sejong_corpus_li if not i[1] == 'Unk']

with open('sejong_corpus_li.pkl', 'wb') as f:
    pickle.dump(sejong_corpus_li, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## 전처리 수행 피드백받고 새로운 corpus 뭉치 생성
- '과연이것도너가분리할수있을까후아하하하하키키키'
- '['과연', '이것', '도너', '가분', '리할수있', '을까', '후', '아하하하', '하키', '키키']

```python
written_eojeol = [(i, 'Eojeol') for i in written_table['Eojeol'].map(
    lambda x : re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', x)) if not len(i) < 2]
colloquial_eojeol = [(i, 'Eojeol') for i in colloquial_table['Eojeol'].map(
    lambda x : re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', x)) if not len(i) < 2]
    
eojeol_corpus_li = list(set(written_eojeol).union(set(colloquial_eojeol)))

with open('eojeol_corpus_li.pkl', 'wb') as f:
    pickle.dump(eojeol_corpus_li, f, protocol=pickle.HIGHEST_PROTOCOL)
```
