#-*- coding:utf-8 -*-

# 1、性能测试函数
# 1.1 引入包
import time
import numpy as np
# 1.2 主函数
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
        print '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)
# 1.3 demo运行函数
def printRunTimeDemo():
    t1 = time.time()
    time.sleep(1)
    printRunTime(t1)
    t1 = time.time()
    time.sleep(1)
    printRunTime(t1, 'name')
    pass


# 2、获取日期list函数
# 2.1 引入包
import pandas as pd
# 2.2 主函数
def getDateRange(start_date, end_date, freq='D'):
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date, freq=freq).values)
    return date_range
# 2.3 demo运行函数
def getDateRangeDemo():
    start_date = '2016-12-01'
    end_date = '2016-12-31'
    getDateRange(start_date, end_date, freq='D')
    # getDateRange(start_date, end_date, freq='M')
    # getDateRange(start_date, end_date, freq='H')
    pass


# 3、排列组合
# 3.1 引入包
from copy import deepcopy
import numpy as np
# 3.2 主函数
class Combine():
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

    def __combineTree(self, root, rest, depth):
        '''
        生成树函数
        '''
        depth += 1
        if depth <= self.tree_q:
            for each in rest:
                next_rest = deepcopy(rest)
                next_root = root + [each]
                next_rest.remove(each)
                self.__combineTree(next_root, next_rest, depth)
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
        self.__combineTree([], self.arr_list, 0)
        if self.reverse:
            self.tree_combine_List = map(lambda x: list(set(self.arr_list).difference(set(x))),self.tree_combine_List)
        return self.tree_combine_List

    def createTree(self):
        '''
        生成组合
        '''
        for each in range(1, self.tree_n + 1):
            self.tree_q = each
            self.__combineTree([], self.arr_list, 0)
        return self.tree_combine_List
# 3.3 demo运行函数
def combineDemo():
    combine = Combine(6)
    combine.CNM(2)
    print combine.tree_combine_List
    combine = Combine(list('abcdef'))
    combine.CNM(2)
    print combine.tree_combine_List


if __name__ == '__main__':
    # printRunTimeDemo()
    combineDemo()
    pass
