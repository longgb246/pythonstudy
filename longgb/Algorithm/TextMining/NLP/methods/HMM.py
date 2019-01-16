# -*- coding:utf-8 -*-
from __future__ import division, print_function
import os
import jieba
import re
import numpy as np
import json
from collections import Counter
from collections import defaultdict


def uppath(n=1):
    if n == 0:
        return os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), (os.pardir + os.sep) * (n - 1) + os.pardir))


epsilon = -3.14e+100


# -----------------------------------------------------
# --    创建语料库
# -----------------------------------------------------
def gen_corpus():
    """ 生成语料库 - 造标记的语料数据。

    读取 264_洛炜言情小说全集83本 小数文本的内容，使用 jieba 进行分词处理，然后标记 BMES。
    """
    # 读取数据
    read_path = os.path.join(uppath(2), 'Data', '264_洛炜言情小说全集83本').decode('utf-8')
    save_path = os.path.join(uppath(2), 'Data', 'corpus')
    dir_list = os.listdir(read_path)
    with open(save_path + os.sep + 'corpus_luowei_novel.txt', 'w') as fw:
        for i, each in enumerate(dir_list):
            print(i + 1, ' : ', each)
            read_file = read_path + os.sep + dir_list[0]
            with open(read_file, 'r') as f:
                content_all = f.readlines()
            content_str = reduce(lambda x, y: x + y, content_all)
            content = re.findall(r'======\*/(.*?)/\*======', content_str, re.S)[0]
            # u'\u3000':空格,  u'\u300c':「,  u'\u300c':」
            content_lcut = jieba.lcut(
                content.decode('gbk').replace(u'\u3000', u'').replace(u'\u300c', u'').replace(u'\u300d', u''))
            # 获得单词的打标
            content_tag = map(lambda x: tag_words(x), content_lcut)
            content_tag_str = '\t'.join(content_tag)
            fw.write(content_tag_str.replace('\n\t', '\n').encode('utf-8'))
            # 特殊字符：u'\u3002':。    u'\u2026':…     u'\uff0c':，    u'\uff01':！    u'\uff1f':？


def tag_words(x):
    """ 单词打标 标记 BMES """
    if len(x) == 1:
        # '\u3000'：全角的空白符， '\xa0'：不间断空白符 &nbsp;
        if x == '\n' or x == ' ' or x == '\t' or x == u'\u3000' or x == u'\xa0':
            follow_str = ''
        else:
            follow_str = '/S'
    elif len(x) == 2:
        follow_str = '/BE'
    elif len(x) > 2:
        follow_str = '/B' + 'M' * (len(x) - 2) + 'E'
    else:
        follow_str = ''
    x_str = x + follow_str
    return x_str


# -----------------------------------------------------
# --    计算初始状态矩阵 InitStatus
# -----------------------------------------------------
def countSB(y):
    """ 统计句首的s、b的出现次数 """
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


def stop_word(x):
    """ 以断句词 。！？ 划分 """
    # 特殊字符：u'\u3002':。    u'\uff01':！    u'\uff1f':？
    x1 = x.split(u'\u3002/S')
    x2 = reduce(lambda m, n: m + n, map(lambda y: y.split(u'\uff01/S'), x1))
    x3 = reduce(lambda m, n: m + n, map(lambda y: y.split(u'\uff1f/S'), x2))
    count_S, count_B = reduce(lambda m, n: (m[0] + n[0], m[1] + n[1]), map(lambda x: countSB(x), x3))
    return count_S, count_B


def cal_InitStatus():
    """ 计算初始矩阵 每个句首的标记 """
    InitStatus = {}
    InitStatus['E'] = -3.14e+100  # 第一个词不可能是 E（end）、M（middle） 的状态
    InitStatus['M'] = -3.14e+100
    read_path = os.path.join(uppath(2), 'Data', 'corpus')
    save_path = os.path.join(uppath(2), 'Data', 'corpus')
    with open(read_path + os.sep + 'corpus_luowei_novel.txt', 'r') as f:
        content = f.readlines()
        count_S, count_B = reduce(lambda m, n: (m[0] + n[0], m[1] + n[1]),
                                  map(lambda x: stop_word(x.decode('utf-8')), content))
        InitStatus['S'] = np.log(count_S / (count_S + count_B))  # p 的 log 值
        InitStatus['B'] = np.log(count_B / (count_S + count_B))
    with open(save_path + os.sep + 'InitStatus.json', 'w') as f:
        json.dump(InitStatus, f)


# -----------------------------------------------------
# --    计算转移概率矩阵 TransProbMatrix
# -----------------------------------------------------
def split_by_slash_trans(y):
    """ 按照斜线分割，取出状态 """
    y_list = y.split('/')
    if len(y_list) <= 1:
        return ''
    else:
        return y_list[1]


def split_by_tab_trans(x):
    """ 按照tab分割 """
    x_list = x.split('\t')
    statusSeries = reduce(lambda n, m: n + m, map(lambda y: split_by_slash_trans(y), x_list))
    return statusSeries


def cal_TransProbMatrix():
    """ 计算状态转移矩阵 """
    TransProbMatrix = {}
    # 'S B M E'   a：表示发生该状态的转移的概率， 0：表示不可能发生转移
    # S -> S B M E
    #      a a 0 0
    # B -> S B M E
    #      0 0 a a
    # M -> S B M E
    #      0 0 a a
    # E -> S B M E
    #      a a 0 0
    # cal S
    read_path = os.path.join(uppath(2), 'Data', 'corpus')
    save_path = os.path.join(uppath(2), 'Data', 'corpus')
    with open(read_path + os.sep + 'corpus_luowei_novel.txt', 'r') as f:
        content = f.readlines()
        print('Start cal TransProbMatrix ...')
        statusSeries = reduce(lambda n, m: n + m, map(lambda x: split_by_tab_trans(x), content))  # 状态序列
        statusSeries_com = [statusSeries[i] + statusSeries[i + 1] for i in range(len(statusSeries) - 1)]  # 状态序列对
        statusSeries_comCount = Counter(statusSeries_com)
        # S -> S B M E
        TransProbMatrix_S = {}
        TransProbMatrix_S['S'] = np.log(
            statusSeries_comCount['SS'] / (statusSeries_comCount['SS'] + statusSeries_comCount['SB']))
        TransProbMatrix_S['B'] = np.log(
            statusSeries_comCount['SB'] / (statusSeries_comCount['SS'] + statusSeries_comCount['SB']))
        TransProbMatrix_S['M'] = epsilon
        TransProbMatrix_S['E'] = epsilon
        TransProbMatrix['S'] = TransProbMatrix_S
        # B -> S B M E
        TransProbMatrix_B = {}
        TransProbMatrix_B['S'] = epsilon
        TransProbMatrix_B['B'] = epsilon
        TransProbMatrix_B['M'] = np.log(
            statusSeries_comCount['BM'] / (statusSeries_comCount['BM'] + statusSeries_comCount['BE']))
        TransProbMatrix_B['E'] = np.log(
            statusSeries_comCount['BE'] / (statusSeries_comCount['BM'] + statusSeries_comCount['BE']))
        TransProbMatrix['B'] = TransProbMatrix_B
        # M -> S B M E
        TransProbMatrix_M = {}
        TransProbMatrix_M['S'] = epsilon
        TransProbMatrix_M['B'] = epsilon
        TransProbMatrix_M['M'] = np.log(
            statusSeries_comCount['MM'] / (statusSeries_comCount['MM'] + statusSeries_comCount['ME']))
        TransProbMatrix_M['E'] = np.log(
            statusSeries_comCount['ME'] / (statusSeries_comCount['MM'] + statusSeries_comCount['ME']))
        TransProbMatrix['M'] = TransProbMatrix_M
        # E -> S B M E
        TransProbMatrix_E = {}
        TransProbMatrix_E['S'] = np.log(
            statusSeries_comCount['ES'] / (statusSeries_comCount['ES'] + statusSeries_comCount['EB']))
        TransProbMatrix_E['B'] = np.log(
            statusSeries_comCount['EB'] / (statusSeries_comCount['ES'] + statusSeries_comCount['EB']))
        TransProbMatrix_E['M'] = epsilon
        TransProbMatrix_E['E'] = epsilon
        TransProbMatrix['E'] = TransProbMatrix_E
    with open(save_path + os.sep + 'TransProbMatrix.json', 'w') as f:
        json.dump(TransProbMatrix, f)


# -----------------------------------------------------
# --    计算发射概率矩阵 EmitProbMatrix
# -----------------------------------------------------
def split_by_slash_emit(y):
    """ 按照斜线分割 """
    y_list = y.split('/')
    if len(y_list) <= 1:
        return ['', '']
    else:
        return y_list


def split_by_tab_emit(x):
    """ 按照tab分割 """
    x_list = x.split('\t')
    wordsSeries, statusSeries = reduce(lambda n, m: [n[0] + m[0], n[1] + m[1]],
                                       map(lambda y: split_by_slash_emit(y), x_list))
    return wordsSeries, statusSeries


def cal_EmitProbMatrix():
    """ 计算 发射矩阵(dict形式)

    A 为当前状态，计算的为 log(P(word|A))，在该状态下出现这个词的概率
    """
    read_path = os.path.join(uppath(2), 'Data', 'corpus')
    save_path = os.path.join(uppath(2), 'Data', 'corpus')
    EmitProbMatrix = defaultdict(lambda: defaultdict(int))
    with open(read_path + os.sep + 'corpus_luowei_novel.txt') as f:
        content = f.readlines()
        wordsSeries, statusSeries = reduce(lambda n, m: [n[0] + m[0], n[1] + m[1]],
                                           map(lambda x: split_by_tab_emit(x.decode('utf-8')), content))  # 获取 单词序列、状态序列
        for status in list('SBME'):
            print('status : ', status)
            thisWordsSeries = []
            for i, thisStatus in enumerate(statusSeries):
                if thisStatus == status:
                    thisWordsSeries.append(wordsSeries[i])
            wordsSeriesCount = Counter(thisWordsSeries)
            wordsSeriesCountAll = np.sum(wordsSeriesCount.values())
            wordsSeriesCountLog = dict(
                map(lambda x: (x[0], np.log(x[1] / wordsSeriesCountAll)), wordsSeriesCount.iteritems()))
            EmitProbMatrix[status] = wordsSeriesCountLog
    with open(save_path + os.sep + 'EmitProbMatrix.json', 'w') as f:
        json.dump(EmitProbMatrix, f)
    # with open(read_path + os.sep + 'EmitProbMatrix.json', 'r') as f:
    #     EmitProbMatrix2 = json.load(f)


# -----------------------------------------------------
# --    中文分词的类
# -----------------------------------------------------
class HMMLgb(object):
    def __init__(self):
        """ 初始化得到 InitStatus，TransProbMatrix，EmitProbMatrix 矩阵 """
        # self.read_path = r'D:\Work\Codes\pythonstudy\longgb\Algorithm\TextMining\Data\corpus'
        self.read_path = os.path.join(uppath(2), 'Data', 'corpus')
        self.save_path = os.path.join(uppath(2), 'Data', 'corpus')
        self.InitStatus = self._load_data('InitStatus.json')
        self.TransProbMatrix = self._load_data('TransProbMatrix.json')
        self.EmitProbMatrix = self._load_data('EmitProbMatrix.json')
        self.statusOrder = list('SBME')
        self.epsilon = epsilon
        self.words = None
        self.lenWords = None
        self.splitWordsCut = None
        self.words_list = []
        self.sens_list = None
        self.statusSeries = ''

    def _load_data(self, file_name):
        """ 读取json文件，读取相应的矩阵信息 """
        with open(self.read_path + os.sep + file_name, 'r') as f:
            data = json.load(f)
        return data

    def _viterbi(self, words_v):
        """ viterbi算法 """
        lenWords = len(words_v)
        # S B M E
        wordWeight = [[self.epsilon] * lenWords for x in range(4)]
        # wordWeight = [[self.epsilon] * lenWords] * 4
        wordPath = [[0] * lenWords for x in range(4)]
        # wordPath = [[0] * lenWords] * 4
        # 初始化 weight 矩阵
        for i_v, each_v in enumerate(self.statusOrder):
            wordWeight[i_v][0] = self.InitStatus[each_v]
        # 1.遍历单词
        for i in range(1, lenWords):
            # 2.遍历当前状态
            for j in range(4):
                # 3.遍历前一个状态
                for k in range(4):
                    try:
                        calEmitProbMatrix = self.EmitProbMatrix[self.statusOrder[j]][words_v[i]]
                    except:
                        # calEmitProbMatrix = -1.6  # 新出现的词，给个概率
                        calEmitProbMatrix = -3e+20  # 新出现的词，给个概率
                    calTransProbMatrix = self.TransProbMatrix[self.statusOrder[k]][self.statusOrder[j]]
                    tmp = wordWeight[k][i - 1] + calTransProbMatrix + calEmitProbMatrix
                    if tmp > wordWeight[j][i]:
                        wordWeight[j][i] = tmp
                        wordPath[j][i] = k
        # 找到开始序列
        startValue = 0
        maxValue = wordWeight[0][lenWords - 1]
        for i in range(4):
            if wordWeight[i][lenWords - 1] > maxValue:
                maxValue = wordWeight[i][lenWords - 1]
                startValue = i
        statusSeries = [self.statusOrder[startValue]]
        # 得到状态序列
        nextValue = wordPath[startValue][lenWords - 1]
        for i in range(1, lenWords - 1)[::-1]:
            statusSeries.append(self.statusOrder[nextValue])
            nextValue = wordPath[nextValue][i]
        statusSeries.append(self.statusOrder[nextValue])
        statusSeries = statusSeries[::-1]
        statusSeries = ''.join(statusSeries)
        return self._splitWords(words_v, statusSeries)

    def _splitWords(self, words_v, statusSeries):
        """ 拆分字串 拆分情况，考虑略繁琐 """
        headIndex = 0
        i = 0
        splitWordsCut = []
        for i, s in enumerate(statusSeries):
            if i < (len(statusSeries) - 1) and \
                    ((statusSeries[i + 1] in ['S', 'B']) or  # 1、下一个单词以 'S', 'B' 开头
                     (self.TransProbMatrix[s][statusSeries[i + 1]] < epsilon / 10.0)):  # 2、下一个与上一个不可能匹配
                splitWordsCut.append(words_v[headIndex:(i + 1)])
                headIndex = i + 1
        if headIndex != (i + 1):
            splitWordsCut.append(words_v[headIndex:(i + 1)])
        return splitWordsCut, statusSeries

    def _batch_cut(self):
        words = self.words
        stop_words_list = [u'\u3002', u'\uff01', u'!', u'\uff1f']
        index_v = 0
        words_list = []
        i = 0
        for i, each_v in enumerate(words):
            if each_v in stop_words_list:
                words_list.append(words[index_v: (i + 1)].
                                  strip().
                                  replace('\n', '').
                                  replace(' ', ''))
                index_v = i + 1
        if index_v != (i + 1):
            words_list.append(words[index_v: (i + 1)].
                              strip().
                              replace('\n', '').
                              replace(' ', ''))
        for each_word in words_list:
            splitWordsCut, statusSeries = self._viterbi(each_word)
            self.words_list.extend(splitWordsCut)
            self.statusSeries += statusSeries
        self.sens_list = words_list

    def lcut(self, words):
        """ 分词 """
        self.words = words
        lenWords = len(self.words)
        if lenWords <= 1:
            self.words_list = self.words
            return self.words
        else:
            self._batch_cut()
            return self.words_list


def main():
    gen_corpus_flag = False  # 是否重新创建 语料库
    cal_InitStatus_flag = False  # 是否重新计算 初始状态矩阵
    cal_TransProbMatrix_flag = False  # 是否重新计算 状态转移矩阵
    cal_EmitProbMatrix_flag = False  # 是否重新计算 发射矩阵

    if gen_corpus_flag:
        gen_corpus()
    if cal_InitStatus_flag:
        cal_InitStatus()
    if cal_TransProbMatrix_flag:
        cal_TransProbMatrix()
    if cal_EmitProbMatrix_flag:
        cal_EmitProbMatrix()

    HMMLgbIns = HMMLgb()
    words = u'''报道称，日本一直拒绝与美国进行双边贸易谈判。此前路透社报道称，日本副首相麻生太郎3月29日
        排除了与美国就双边贸易协议举行会谈的可能性，他强调这样的协议对日本没有好处。麻生表示，日本不应加入双边
        自贸协定（FTA）谈判来换取豁免钢铝关税，“当两个国家谈判时，强者恒强，只会给日本带来不必要的痛苦。”菅义伟
        也提醒称，关于美国重新加入TPP的任何谈判都可能面临挑战，因为11个签署国可能并不愿再与美国重启此前相关问题
        的谈判。菅义伟称，TPP协议就像“易碎的玻璃”，“要把一部分提出来重新谈判，将会非常困难。”
        '''
    # words = u'''这是一个测试！'''

    aa = HMMLgbIns.lcut(words)
    tmp_line = []
    line_num = 8
    for i, each in enumerate(aa):
        if ((i > 0) and (i % line_num == 0)) or ((len(aa) <= line_num) and (i == (len(aa) - 1))):
            tmp_line.append(each)
            print('  '.join(tmp_line))
            tmp_line = []
        else:
            tmp_line.append(each)
    # HMMLgbIns.statusSeries


if __name__ == '__main__':
    main()
    # 最大熵马尔可夫模型 MEMM
