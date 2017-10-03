#-*- coding:utf-8 -*-
from __future__ import division
import os
import jieba
import re
import numpy as np
import json
from collections import Counter
from collections import defaultdict


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
# --    计算初始状态矩阵 InitStatus
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


# -----------------------------------------------------
# --    计算转移概率矩阵 TransProbMatrix
# -----------------------------------------------------
def splitBySlashTrans(y):
    '''
    按照斜线分割
    '''
    y_list = y.split('/')
    if len(y_list) <= 1:
        return ''
    else:
        return y_list[1]


def splitByTabTrans(x):
    '''
    按照tab分割
    '''
    x_list = x.split('\t')
    statusSeries = reduce(lambda n,m: n+m, map(lambda y: splitBySlashTrans(y), x_list))
    return statusSeries


def calTransProbMatrix():
    '''
    计算状态转移矩阵
    '''
    TransProbMatrix = {}
    # 'S B M E'
    # S -> S B M E
    #      a a 0 0
    # B -> S B M E
    #      0 0 a a
    # M -> S B M E
    #      0 0 a a
    # E -> S B M E
    #      a a 0 0
    # cal S
    read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    save_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    with open(read_path + os.sep + 'corpus_luowei_novel.txt', 'r') as f:
        content = f.readlines()
        print 'Start cal TransProbMatrix ...'
        statusSeries = reduce(lambda n,m: n+m, map(lambda x: splitByTabTrans(x), content))   # 耗时
        statusSeries_com = [statusSeries[i]+statusSeries[i+1] for i in range(len(statusSeries)-1)]
        statusSeries_comCount = Counter(statusSeries_com)
        # S -> S B M E
        TransProbMatrix_S = {}
        TransProbMatrix_S['S'] = np.log(statusSeries_comCount['SS']/(statusSeries_comCount['SS']+statusSeries_comCount['SB']))
        TransProbMatrix_S['B'] = np.log(statusSeries_comCount['SB']/(statusSeries_comCount['SS']+statusSeries_comCount['SB']))
        TransProbMatrix_S['M'] = -3.14e+100
        TransProbMatrix_S['E'] = -3.14e+100
        TransProbMatrix['S'] = TransProbMatrix_S
        # B -> S B M E
        TransProbMatrix_B = {}
        TransProbMatrix_B['S'] = -3.14e+100
        TransProbMatrix_B['B'] = -3.14e+100
        TransProbMatrix_B['M'] = np.log(statusSeries_comCount['BM']/(statusSeries_comCount['BM']+statusSeries_comCount['BE']))
        TransProbMatrix_B['E'] = np.log(statusSeries_comCount['BE']/(statusSeries_comCount['BM']+statusSeries_comCount['BE']))
        TransProbMatrix['B'] = TransProbMatrix_B
        # M -> S B M E
        TransProbMatrix_M = {}
        TransProbMatrix_M['S'] = -3.14e+100
        TransProbMatrix_M['B'] = -3.14e+100
        TransProbMatrix_M['M'] = np.log(statusSeries_comCount['MM']/(statusSeries_comCount['MM']+statusSeries_comCount['ME']))
        TransProbMatrix_M['E'] = np.log(statusSeries_comCount['ME']/(statusSeries_comCount['MM']+statusSeries_comCount['ME']))
        TransProbMatrix['M'] = TransProbMatrix_M
        # E -> S B M E
        TransProbMatrix_E = {}
        TransProbMatrix_E['S'] = np.log(statusSeries_comCount['ES']/(statusSeries_comCount['ES']+statusSeries_comCount['EB']))
        TransProbMatrix_E['B'] = np.log(statusSeries_comCount['EB']/(statusSeries_comCount['ES']+statusSeries_comCount['EB']))
        TransProbMatrix_E['M'] = -3.14e+100
        TransProbMatrix_E['E'] = -3.14e+100
        TransProbMatrix['E'] = TransProbMatrix_E
    with open(save_path + os.sep + 'TransProbMatrix.json', 'w') as f:
        json.dump(TransProbMatrix, f)


# -----------------------------------------------------
# --    计算发射概率矩阵 EmitProbMatrix
# -----------------------------------------------------
def splitBySlashEmit(y):
    '''
    按照斜线分割
    '''
    y_list = y.split('/')
    if len(y_list) <= 1:
        return ['','']
    else:
        return y_list


def splitByTabEmit(x):
    '''
    按照tab分割
    '''
    x_list = x.split('\t')
    wordsSeries, statusSeries = reduce(lambda n,m: [n[0]+m[0], n[1]+m[1]], map(lambda y: splitBySlashEmit(y), x_list))
    return wordsSeries, statusSeries


def calEmitProbMatrix():
    read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    save_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
    EmitProbMatrix = defaultdict(lambda :defaultdict(int))
    with open(read_path + os.sep + 'corpus_luowei_novel.txt') as f:
        content = f.readlines()
        wordsSeries, statusSeries = reduce(lambda n,m: [n[0]+m[0], n[1]+m[1]], map(lambda x: splitByTabEmit(x.decode('utf-8')), content))
        for status in list('SBME'):
            print 'status : ', status
            thisWordsSeries = []
            for i, thisStatus in enumerate(statusSeries):
                if thisStatus == status:
                    thisWordsSeries.append(wordsSeries[i])
            wordsSeriesCount = Counter(thisWordsSeries)
            wordsSeriesCountAll = np.sum(wordsSeriesCount.values())
            wordsSeriesCountLog = dict(map(lambda x: (x[0], np.log(x[1]/wordsSeriesCountAll)),wordsSeriesCount.iteritems()))
            EmitProbMatrix[status] = wordsSeriesCountLog
    with open(save_path + os.sep + 'EmitProbMatrix.json', 'w') as f:
        json.dump(EmitProbMatrix, f)
    # with open(read_path + os.sep + 'EmitProbMatrix.json', 'r') as f:
    #     EmitProbMatrix2 = json.load(f)


epsilon = -3.14e+100


# -----------------------------------------------------
# --    中文分词的类
# -----------------------------------------------------
class HMMLgb():
    def __init__(self):
        '''
        初始化得到 InitStatus，TransProbMatrix，EmitProbMatrix 矩阵
        '''
        self.read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
        self.save_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
        self.InitStatus = self.__loadData('InitStatus.json')
        self.TransProbMatrix = self.__loadData('TransProbMatrix.json')
        self.EmitProbMatrix = self.__loadData('EmitProbMatrix.json')
        self.statusOrder = list('SBME')
        self.epsilon = -3.14e+100

    def __loadData(self, file):
        '''
        读取json文件，读取相应的矩阵信息
        '''
        with open(self.read_path + os.sep + file, 'r') as f:
            data = json.load(f)
        return data

    def __viterbi(self):
        '''
        viterbi算法
        '''
        # S B M E
        self.wordWeight = [[self.epsilon]*self.lenWords for x in range(4)]
        self.wordPath = [[0]*self.lenWords for x in range(4)]
        # 初始化 weight 矩阵
        for i, each in enumerate(self.statusOrder):
            self.wordWeight[i][0] = self.InitStatus[each]
        # 1.遍历单词
        for i in range(1, self.lenWords):
            # 2.遍历当前状态
            for j in range(4):
                # 3.遍历前一个状态
                for k in range(4):
                    try:
                        calEmitProbMatrix = self.EmitProbMatrix[self.statusOrder[j]][self.words[i]]
                    except:
                        calEmitProbMatrix = -1.6
                    calTransProbMatrix = self.TransProbMatrix[self.statusOrder[k]][self.statusOrder[j]]
                    tmp = self.wordWeight[k][i-1] + calTransProbMatrix + calEmitProbMatrix
                    if tmp > self.wordWeight[j][i]:
                        self.wordWeight[j][i] = tmp
                        self.wordPath[j][i] = k
        # 找到开始序列
        startValue = 0
        maxValue = self.wordWeight[0][self.lenWords-1]
        for i in range(4):
            if self.wordWeight[i][self.lenWords-1] > maxValue:
                maxValue = self.wordWeight[i][self.lenWords-1]
                startValue = i
        statusSeries = [self.statusOrder[startValue]]
        # 得到状态序列
        nextValue = startValue
        for i in range(1, self.lenWords-1)[::-1]:
            statusSeries.append(self.statusOrder[nextValue])
            nextValue = self.wordPath[nextValue][i]
        statusSeries.append(self.statusOrder[nextValue])
        statusSeries = statusSeries[::-1]
        self.statusSeries = ''.join(statusSeries)
        self.__splitWords()

    def __splitWords(self):
        '''
        拆分字串
        '''
        # 拆分
        headIndex = 0
        splitWordsCut = []
        for i in range(len(self.statusSeries)):
            thisSplitWordsCut = []
            # 当前结束情况
            situation1 = self.statusSeries[headIndex] == 'S' and self.statusSeries[i] == 'S'
            situation7 = self.statusSeries[headIndex] == 'B' and self.statusSeries[i] == 'E'
            # 下一个结束情况
            situation2 = self.statusSeries[headIndex] == 'S' and self.statusSeries[i] == 'B'
            situation3 = self.statusSeries[headIndex] == 'S' and self.statusSeries[i] == 'E'
            situation4 = self.statusSeries[headIndex] == 'S' and self.statusSeries[i] == 'M'

            situation5 = self.statusSeries[headIndex] == 'B' and self.statusSeries[i] == 'S'
            situation6 = self.statusSeries[headIndex] == 'B' and self.statusSeries[i] == 'B' and headIndex!=i

            situation8 = self.statusSeries[headIndex] == 'E' and self.statusSeries[i] == 'S'
            situation9 = self.statusSeries[headIndex] == 'E' and self.statusSeries[i] == 'B'
            situation10 = self.statusSeries[headIndex] == 'E' and self.statusSeries[i] == 'E' and headIndex!=i
            situation11 = self.statusSeries[headIndex] == 'E' and self.statusSeries[i] == 'M'

            situation12 = self.statusSeries[headIndex] == 'M' and self.statusSeries[i] == 'S'
            situation13 = self.statusSeries[headIndex] == 'M' and self.statusSeries[i] == 'B'

            if situation1 or situation7:
                thisSplitWordsCut = self.words[headIndex:(i+1)]
                splitWordsCut.append(thisSplitWordsCut)
                if i < (self.lenWords-1):
                    headIndex = i+1
            elif situation2 or situation4 or situation5 or situation6 or situation3 \
                    or situation8 or situation9 or situation10 or situation11 or situation12 or situation13:
                thisSplitWordsCut = self.words[headIndex:i]
                splitWordsCut.append(thisSplitWordsCut)
                headIndex = i
                if headIndex == (len(self.statusSeries)-1):
                    thisSplitWordsCut = self.words[headIndex:]
                    splitWordsCut.append(thisSplitWordsCut)
            if i == (len(self.statusSeries)-1) and thisSplitWordsCut==[]:
                    thisSplitWordsCut = self.words[headIndex:]
                    splitWordsCut.append(thisSplitWordsCut)
        self.splitWordsCut = splitWordsCut

    def lcut(self, words):
        '''
        分词
        '''
        self.words = words
        self.lenWords = len(self.words)
        if self.lenWords <= 1:
            self.splitWordsCut = self.words
            return self.words
        else:
            self.__viterbi()
            # print self.words
            # print self.statusSeries
            return self.splitWordsCut


if __name__ == '__main__':
    HMMLgbIns = HMMLgb()
    words = u'我这是一个测试事件。'

    aa = HMMLgbIns.lcut(words)
    for each in aa:
        print each
    # HMMLgbIns.statusSeries

