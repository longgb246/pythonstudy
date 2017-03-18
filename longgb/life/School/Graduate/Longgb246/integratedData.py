#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import re
import time


def printRunTime(t1, name=""):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if name != "":
        name = " ( " + name + " )"
    if hor_d >0:
        print ' [ Run Time ]{3} is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print ' [ Run Time ]{2} is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


def calImportant(sent, extract_tags, topK):
    sent_split = jieba.lcut(sent, cut_all=False)
    count = 0
    for each in sent_split:
        if each in extract_tags:
            count += 1
    important = count**2/topK
    return important


def contentAbstract(str_tmp):
    fenju_re = '\.|。|!|！|\?|？'
    fenju_list = re.split(fenju_re ,str_tmp)
    topK = 10
    extract_tags = jieba.analyse.extract_tags(str_tmp, topK=topK)
    fenju_list_map = map(lambda sent: calImportant(sent, extract_tags, topK), fenju_list)
    fenju_argsort = np.argsort(fenju_list_map)
    str_abstract = reduce(lambda x, y: x + " " + y,[fenju_list[x] for x in fenju_argsort[-5:]])
    return str_abstract


def parseTextContent(file_dir, com_code):
    pd_list = []
    for line in open(file_dir):
        tmp = line.split('|')
        if len(tmp) > 4:
            str_tmp = reduce(lambda x, y: x + y, tmp[3:])
        else:
            str_tmp = tmp[3]
        str_abstract = contentAbstract(str_tmp)
        tmp_list = [tmp[0][:10], tmp[2], str_abstract]
        pd_list.append(tmp_list)
    data = pd.DataFrame(pd_list, columns=['date', 'title', 'cont_abstract'])
    data['com_code'] = com_code
    data['date'] = pd.to_datetime(data['date'])
    return data


def readTextContent(each_dir):
    list_dir = read_path + os.sep + each_dir
    list_dir_name = os.listdir(list_dir)
    data = pd.DataFrame()
    for i, each in enumerate(list_dir_name):
        print '     [ subDir ]: ', each,
        passNum = 0
        if passNum != 0:
            data = pd.read_table(save_path + os.sep + '{0}_tmp_{1}.txt'.format(each_dir, passNum), sep='|')
        if i <= passNum:
            pass
        else:
            try:
                t1 = time.time()
                list_dir_2 = list_dir + os.sep + each
                com_code = each
                if os.path.isdir(list_dir_2):
                    file_dir = list_dir_2 + os.sep + 'content.txt'
                    if os.path.exists(file_dir):
                        tmp_data = parseTextContent(file_dir, com_code)
                        data = pd.concat([data, tmp_data])
                if divmod(i + 1,100)[1] == 0:
                    data.to_csv(save_path + os.sep + '{0}_tmp_{1}.txt'.format(each_dir, i + 1), index=False, sep='|')
                printRunTime(t1)
            except:
                print '[ Warning !] There is Error !'
                with open(save_path + os.sep + 'error_com.log', 'a') as f:
                    f.write('     [ subDir ]: {0}'.format(each))
                    f.write('\n')
    return data


def readFinance():
    file = read_path + os.sep + 'Finance'
    read_finance = pd.DataFrame()
    for read_dir in read_dirs:
        read_finance_path = file + os.sep + 'stock_price_{0}.csv'.format(read_dir)
        tmp_finance = pd.read_csv(read_finance_path)
        tmp_finance = tmp_finance.loc[:,['date', 'open', 'close', 'code']]
        read_finance = pd.concat([read_finance, tmp_finance])
    read_finance['change'] = read_finance['close'] - read_finance['open']
    read_finance['change_flag'] = map(lambda x: 1 if x > 0 else 0,read_finance['change'].values)
    read_finance = read_finance.rename(columns={'code':'com_code'})
    read_finance['date'] = pd.to_datetime(read_finance['date'])
    return read_finance


# =================================================================
# =                             参数配置                          =
# =================================================================
read_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
save_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Now_Data\integratedData'

read_dirs = ['sha', 'shb', 'szb', 'szcy', 'szz', 'szzx']

isSave = True                           # 是否中间保存解析的文件


if __name__ == '__main__':
    read_finance = readFinance()
    print 'read_finance...'
    if isSave:
        read_finance.to_csv(save_path + os.sep + 'read_finance.csv', index=False)
    text_data = pd.DataFrame()
    for each_dir in read_dirs:
        print '[ Dir ]: ',each_dir
        try:
            data = readTextContent(each_dir)
            if isSave:
                data.to_csv(save_path + os.sep + '{0}.txt'.format(each_dir), index=False, sep='|')
            combine = data.merge(read_finance, on=['date', 'com_code'])
            if isSave:
                combine.to_csv(save_path + os.sep + 'combine_{0}.txt'.format(each_dir), index=False, sep='|')
        except:
            print '[ Warning !] There is Error !'
            with open(save_path + os.sep + 'error_com.log', 'a') as f:
                f.write('[ Dir ]: {0}'.format(each_dir))
                f.write('\n')
    pass

