#-*- coding:utf-8 -*-

# Import Library
import numpy as np
import pandas as pd
import json
import os

# Import custom .py file
from ksenticnet_kaist import *

# Data Load
def load_dataset(path):
    '''
    path : 'raw_data_nsmx.csv' 파일의 path가 파일명까지 입력된 str
    '''
    df = pd.read_csv(path)
    return df
  
def load_senti_dict(path):
    '''
    path : '감정단어사전0603.xlsx' 파일의 path가 파일명까지 입력된 str
    '''
    ksenticnet = get_senticnet()
    sentiwordknu = pd.read_excel('')
    
def main():
    # Load Data
    df = load_dataset(path=)
    # Load Senti Dictionary
    ksenticnet, sentiwordknu = load_senti_dict(path)
    '''
    이곳에 라벨링 수행 작업 기입
    '''
    
    
if __name__ == '__main__':
    main()
