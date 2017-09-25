#-*- coding:utf-8 -*-
from __future__ import division
import os
import jieba
import re
import numpy as np
import json


# -----------------------------------------------------
# --    创建语料库
# -----------------------------------------------------
def genCorpus():
    '''
    生成语料库。
    '''
    # 读取数据
    read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\264_洛炜言情小说全集83本'.decode('utf-8')
    save_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    dir_list = os.listdir(read_path)
    with open(save_path + os.sep + 'corpus_luowei_novel.txt', 'w') as fw:
        for i, each in enumerate(dir_list):
            print i+1, ' : ', each
            read_file = read_path + os.sep + dir_list[0]
            with open(read_file, 'r') as f:
                content_all = f.readlines()
            content_str = reduce(lambda x, y: x+y, content_all)
            content = re.findall(r'======\*/(.*?)/\*======', content_str, re.S)[0]
            # u'\u3000':空格,  u'\u300c':「,  u'\u300c':」
            content_lcut = jieba.lcut(content.decode('gbk').replace(u'\u3000', u'').replace(u'\u300c', u'').replace(u'\u300d', u''))
            # 获得单词的打标
            content_tag = map(lambda x: tagWords(x), content_lcut)
            content_tag_str = '\t'.join(content_tag)
            fw.write(content_tag_str.replace('\n\t','\n').encode('utf-8'))
            # 特殊字符：u'\u3002':。    u'\u2026':…     u'\uff0c':，    u'\uff01':！    u'\uff1f':？


def tagWords(x):
    '''
    单词打标
    '''
    if len(x) == 1:
        # '\u3000'：全角的空白符， '\xa0'：不间断空白符 &nbsp;
        if x == '\n' or x == ' ' or x=='\t' or x==u'\u3000' or x==u'\xa0':
            follow_str = ''
        else:
            follow_str = '/S'
    elif len(x) == 2:
        follow_str = '/BE'
    elif len(x) > 2:
        follow_str = '/B' + 'M'*(len(x)-2) + 'E'
    else:
        follow_str = ''
    x_str = x + follow_str
    return x_str


# -----------------------------------------------------
# --    计算初始矩阵
# -----------------------------------------------------
def countSB(y):
    '''
    统计句首的s、b的出现次数
    '''
    y_list = y.split('\t')
    count_S = 0
    count_B = 0
    if len(y_list) <= 1:
        pass
    else:
        if 'S' in y_list[0]:
            count_S += 1
        elif 'B' in y_list[0]:
            count_B += 1
        elif 'S' in y_list[1]:
            count_S += 1
        elif 'B' in y_list[1]:
            count_B += 1
        else:
            pass
    return count_S, count_B


def stopWord(x):
    '''
    以断句词划分
    '''
    # 特殊字符：u'\u3002':。    u'\uff01':！    u'\uff1f':？
    x1 = x.split(u'\u3002/S')
    x2 = reduce(lambda m, n: m + n, map(lambda y: y.split(u'\uff01/S'), x1))
    x3 = reduce(lambda m, n: m + n, map(lambda y: y.split(u'\uff1f/S'), x2))
    count_S, count_B = reduce(lambda m,n: (m[0]+n[0], m[1]+n[1]) ,map(lambda x: countSB(x), x3))
    return count_S, count_B


def calInitStatus():
    '''
    计算初始矩阵
    '''
    InitStatus = {}
    InitStatus['E'] = -3.14e+100
    InitStatus['M'] = -3.14e+100
    read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    save_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    with open(read_path + os.sep + 'corpus_luowei_novel.txt', 'r') as f:
        content = f.readlines()
        count_S, count_B = reduce(lambda m,n: (m[0]+n[0], m[1]+n[1]), map(lambda x: stopWord(x.decode('utf-8')), content))
        InitStatus['S'] = np.log(count_S/(count_S+count_B))
        InitStatus['B'] = np.log(count_B/(count_S+count_B))
    with open(save_path + os.sep + 'InitStatus.json', 'w') as f:
        json.dump(InitStatus, f)


if __name__ == '__main__':


    # with open(read_path + os.sep + 'InitStatus.json', 'r') as f:
    #     InitStatus2 = json.load(f)
    pass
