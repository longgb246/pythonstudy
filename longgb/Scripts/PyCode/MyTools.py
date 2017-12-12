#-*- coding:utf-8 -*-
# ---------------------------------------------------------------------------
# 2.1 引入包
import time
import numpy as np
import datetime
import os
# 2.2 主类
class myTools(object):
    @staticmethod
    def runTime(t1, name="", is_print=True):
        '''
        性能测试函数，测试运行时间。
        :param is_print: True。是否打印
        :return str 打印的字符串内容
        ex：
        t1 = time.time()
        测试的func
        runTime(t1, 'name')
        '''
        d = time.time() - t1
        min_d = np.floor(d / 60)
        sec_d = d % 60
        hor_d = np.floor(min_d / 60)
        if name != "":
            name = " ( " + name + " )"
        if hor_d >0:
            v_str = '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
        else:
            v_str = '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)
        if is_print:
            print(v_str)
        return v_str
    @staticmethod
    def mkdir(path, trash=False, clear=True):
        '''
        创建文件夹，
        :param trash: True， 表示，如果存在该文件夹，1、将该文件夹重命名为 .Trash 文件夹 2、在建立该文件夹
        '''
        if not os.path.exists(path):
            os.mkdir(path)
        elif trash:
            os.system('''cp -rf {0}  .Trash '''.format(path))
            os.system('''rm -rf {0}  '''.format(path))
            os.mkdir(path)
        if clear:
            try:
                os.system('''rm -rf .Trash  ''')
            except:
                pass
    @staticmethod
    def cp(from_path, to_path, is_dir=False):
        '''
        复制文件
        :param from_path: 原文件
        :param to_path: 复制后的文件
        '''
        if is_dir:
            os.system(''' cp -rf {0} {1} ;'''.format(from_path, to_path))
        else:
            os.system(''' cp {0} {1} ;'''.format(from_path, to_path))
    @staticmethod
    def dateRange(start_date, end_date):
        '''
        返回日期函数。统一格式，%Y-%m-%d。
        :param start_date: '2017-10-01' 开始日期
        :param end_date: '2017-11-01' 结束日期
        :return: 日期 list
        '''
        start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d'),range((end_date_dt - start_date_dt).days))
        return date_range


# ---------------------------------------------------------------------------
# 2、排列组合
# 2.1 引入包
from copy import deepcopy
import numpy as np
# 2.2 主类
class Combine(object):
    def __init__(self, arr_list=['0']):
        '''
        生成排列组合
        '''
        self.all_Tree = []
        self.tree_arrange_List = []         # 排列
        self.tree_combine_List = []         # 组合
        self.tree_q = 0
        self.tree_n = len(arr_list) if type(arr_list)==list else arr_list
        self.arr_list = arr_list if type(arr_list)==list else range(self.tree_n)
        self.reverse = False

    def _combineTree(self, root, rest, depth):
        '''
        生成树函数
        '''
        depth += 1
        if depth <= self.tree_q:
            for each in rest:
                next_rest = deepcopy(rest)
                next_root = root + [each]
                next_rest.remove(each)
                self._combineTree(next_root, next_rest, depth)
        else:
            root = sorted(root)
            if root not in self.tree_combine_List:
                self.tree_combine_List.append(root)

    def CNM(self, m=0):
        '''
        C N 取 M 个的组合
        '''
        if m > np.floor(len(self.arr_list)/2):
            m = len(self.arr_list) - m
            self.reverse = True
        self.tree_q = m
        self._combineTree([], self.arr_list, 0)
        if self.reverse:
            self.tree_combine_List = map(lambda x: list(set(self.arr_list).difference(set(x))),self.tree_combine_List)
        return self.tree_combine_List

    def createTree(self):
        '''
        生成组合
        '''
        for each in range(1, self.tree_n + 1):
            self.tree_q = each
            self._combineTree([], self.arr_list, 0)
        return self.tree_combine_List
# 2.3 demo运行函数
def combineDemo():
    combine = Combine(6)
    combine.CNM(2)
    print combine.tree_combine_List
    combine = Combine(list('abcdef'))
    combine.CNM(2)
    print combine.tree_combine_List


# ---------------------------------------------------------------------------
# 3、numpy的方便处理的包
# 3.1 引入包
import numpy as np
# 3.2 主类
class npl(object):
    '''
    numpy local method : npl
    '''
    @staticmethod
    def loc(np_array, cols):
        '''
        按照cols的布尔序列选出array的某些行
        cols 为 [True, False] 组成的 list 或者 np.array
        使用方法如下：
        loc(test, test[:,-1] == 1)  # 抽出test最后一列为1的np.array
        '''
        return np_array[np.where(cols)[0], :]
    @staticmethod
    def sort(np_array, cols, ascending=[]):
        '''
        按cols从小到大排列array的行数据
        cols 为 [3, 1] 数字组成的 list
        使用方法如下：
        sort(test, [3, 1])           # 按照test的第4列，第2列从小到大排序
        '''
        if ascending==[]:
            sort_data = np_array[:, cols[::-1]].T
        else:
            sort_data = ([map(lambda x: 1 if x else -1, ascending[::-1])] * np_array[:, cols[::-1]]).T
        return np_array[np.lexsort(sort_data), :]
    @staticmethod
    def drop_duplicates(np_array):
        '''
        去重
        使用方法如下：
        drop_duplicates(test)           # 去除test的重复的行
        '''
        return np.array(list(set([tuple(x) for x in np_array])))
# 3.3 demo运行函数
def nplDemo():
    def get_data():
        index_data = range(8)
        sales = np.random.rand(8) * 8
        std = np.random.rand(8) * 2
        labels = [0, 1, 0, 1, 2, 0, 2, 1]
        data = np.array(zip(index_data, sales, std, labels))
        return data
    data = get_data()
    npl.loc(data, data[:,-1] == 1)                      # 取出最后一列为1的行
    npl.sort(data, [1, 2], ascending=[True, False])     # 按照第0列升序，第1列降序
    npl.drop_duplicates(data)                           # data数据去重，以每行数据为粒度


# ---------------------------------------------------------------------------
# 4、pandas的方便处理的包
# 4.1 引入包
import pandas as pd
# 4.2 主类
class pdl(object):
    @staticmethod
    def dateRange(start_date, end_date, freq='D'):
        '''
        返回日期函数。
        :param start_date: '2017-10-01' 开始日期
        :param end_date: '2017-11-01' 结束日期
        :param freq: 间隔频率
        :return: 日期 list
        ex:
        getDateRange(start_date, end_date, freq='D')
        getDateRange(start_date, end_date, freq='M')
        getDateRange(start_date, end_date, freq='H')
        '''
        date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date, freq=freq).values)
        return date_range
    @staticmethod
    def tranNum2Str(data, cols):
        '''
        讲num类型转化成str类型
        '''
        for col in cols:
            data[col] = data.loc[:,[col]].applymap(str)
        return data
    @staticmethod
    def tranStr2Num(data, cols):
        '''
        讲str类型转化成num类型
        '''
        for col in cols:
            data[col] = data.loc[:,[col]].applymap(float)
        return data
    @staticmethod
    def tranStr2Int(data, cols):
        '''
        讲str类型转化成int类型
        '''
        for col in cols:
            data[col] = data.loc[:,[col]].applymap(int)
        return data

