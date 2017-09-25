#-*- coding:utf-8 -*-
from __future__ import division
import os
import jieba
import re


def genCorpus():
    '''
    生成语料库
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
            content_lcut = jieba.lcut(content.decode('gbk').replace(u'\u3000', u''))
            # 获得单词的打标
            content_tag = map(lambda x: tagWords(x), content_lcut)
            content_tag_str = '\t'.join(content_tag)
            fw.write(content_tag_str.encode('utf-8'))


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


if __name__ == '__main__':

    pass

