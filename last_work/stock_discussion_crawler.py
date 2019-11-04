#-*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pickle

def crawling(code, numpage):
    b = []
    code = ':0>6s'.format(code)
    numpage = int(numpage)
    for i in range(numpage + 1):
        url = 'https://finance.naver.com/item/board.nhn?code={}&page={}'.format(code, i)
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        a = soup.find_all('td', {'class':'title'})
        for i in a:
            b.append(i.find('a')['title'])
    return b, code, numpage

def main():
    code = input('주식의 종목 코드 6자리를 입력해주세요.')
    numpage = input('몇 페이지를 볼 까요?')
    b = crawling(code, numpage)
    return b
    
if __name__ == '__main__':
    b, code, numpage = main()
    with open('{}_stock_discussion_{}pages'.format(code, numpage), 'w', encoding='utf-8') as f:
        pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
