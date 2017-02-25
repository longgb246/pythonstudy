#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import jieba

jieba_words = r'F:\WorkSpace\pythonstudy\longgb\life\School\Graduate\Longgb246\jieba_words.txt'

def test():
    jieba.load_userdict(jieba_words)

    # 1、有问题！！
    text = '和平主义。'
    jieba.add_word('和平', freq=100000)
    jieba.suggest_freq(('和','平','主','义'), True)
    a = list(jieba.cut(text, HMM=False))
    a = jieba.lcut(text, HMM=False)
    print '\\'.join(a)


    # 2、有问题！！
    text4 = '乒乓球拍卖完了'
    jieba.add_word('乒乓球拍')
    jieba.suggest_freq(('拍', '卖'), True)
    aaa = jieba.lcut(text4, HMM=False)
    print '\\'.join(aaa)

    text2 = '将军任命了一名中将。'
    jieba.add_word('中将')
    aa = jieba.lcut(text2)
    print '\\'.join(aa)
    text3 = '产量三年中将增长两倍'
    jieba.suggest_freq(('中', '将'), True)
    aaa = jieba.lcut(text3)
    print '\\'.join(aaa)

    text4 = '请把手拿开'
    jieba.suggest_freq(('把','手'), True)
    aaa = jieba.lcut(text4)
    print '\\'.join(aaa)
    text4 = '这个门把手坏了'
    aaa = jieba.lcut(text4)
    print '\\'.join(aaa)






if __name__ == "__main__":
    pass

