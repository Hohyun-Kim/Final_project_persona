## 지난 19.09~19.10까지 한 작업들을 정리 및 기입합니다.

#### `nsmc_crawler.py`
- 2019.09.29
    - create version 0.0
    - 장시간 지속될 시 `ConnectionError: ('Connection aborted.', OSError("(10053, 'WSAECONNABORTED')",))` 발생
    - `UnicodeEncodeError: 'utf-8' codec can't encode character '\udc3d' in position 141: surrogates not allowed`
      - 위의 에러 발생 시, `s[:s.find('\udc3d')] +  s[s.find('\udc3d') + len('\udc3d'):]`와 같이 해결
    - 특정 파일 별 osError로 중단되는 현상 발생
    - MovieName 이 첫 영화 이름으로 저장되는 오류 존재
    - `MovieCode`, `MovieComeOut`, `MovieType`, `ReviewCreateTime`
      - 위 네 개 feature를 추출함에 있어서 특정 영화가 위 순서대로 지켜지지 않는 경우 발생
- 2019.11.04
    - version 0.0 git에 upload
