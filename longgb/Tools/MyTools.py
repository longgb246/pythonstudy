# -*- coding:utf-8 -*-
# ---------------------------------------------------------------------------
# 2.1 引入包
import time
import math
import datetime
import os
from dateutil.parser import parse


# 2.2 主类
class MyTools(object):

    @staticmethod
    def run_time(t1, name="", is_print=True):
        """性能测试函数，测试运行时间。

        :param t1: 设置断点的时间
        :param name: 设置打印的名称
        :param is_print: True, 是否打印
        :return: 打印的字符串内容

        Example
        ----------
        >>> t1 = time.time()
        # 测试的func
        >>> MyTools.run_time(t1, 'name')
        """
        d = time.time() - t1
        min_d = math.floor(d / 60)
        sec_d = d % 60
        hor_d = math.floor(min_d / 60)
        if name != "":
            name = " ( " + name + " )"
        if hor_d > 0:
            v_str = '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
        else:
            v_str = '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)
        if is_print:
            print(v_str)
        return v_str

    @staticmethod
    def mkdir(path, trash=False, clear=True):
        """创建文件夹

        :param path:
        :param trash:  True， 表示，如果存在该文件夹，1、将该文件夹重命名为 .Trash 文件夹 2、在建立该文件夹
        :param clear:
        :return:
        """
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
        """复制文件（弃用 - 待修改）

        :param from_path: 原文件
        :param to_path: 复制后的文件
        """
        if is_dir:
            os.system(''' cp -rf {0} {1} ;'''.format(from_path, to_path))
        else:
            os.system(''' cp {0} {1} ;'''.format(from_path, to_path))

    @staticmethod
    def date_range(start_date, end_date):
        """Specifies the start date and end date to get a date list. Uniform format, %Y-%m-%d.

        :param str start_date: start date, include
        :param str end_date: end date, not include
        :return: ( list<string> ) date list

        Example
        ----------
        >>> MyTools.date_range('2017-10-01', '2017-10-04')
        ['2017-10-01', '2017-10-02', '2017-10-03']
        """
        # start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        # date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d'),
        #                  range((end_date_dt - start_date_dt).days))
        start_date_dt = parse(start_date)
        end_date_dt = parse(end_date)
        date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d'),
                         range((end_date_dt - start_date_dt).days))
        return date_range

    @staticmethod
    def date_calculate(start_date, cal_date=0):
        """From the start date to a certain direction to get a date list.

        :param start_date: Start date to calculate
        :param cal_date: From the start date to a certain direction.
        :return: list

        Example
        ----------
        >>> MyTools.date_calculate('2017-03-04', 3)
        ['2017-03-04', '2017-03-05', '2017-03-06', '2017-03-07']
        >>> MyTools.date_calculate('2017-03-04', 0)
        ['2017-03-04']
        >>> MyTools.date_calculate('2017-03-04', -3)
        ['2017-03-01', '2017-03-02', '2017-03-03', '2017-03-04']
        """
        start_date_dt = parse(start_date)
        end_date_dt = start_date_dt + datetime.timedelta(cal_date)
        min_date = min(start_date_dt, end_date_dt)
        max_date = max(start_date_dt, end_date_dt)
        date_range = map(lambda x: (min_date + datetime.timedelta(x)).strftime('%Y-%m-%d'),
                         range((max_date - min_date).days + 1))
        return date_range


# ---------------------------------------------------------------------------
# 2、排列组合
# 2.1 引入包
from copy import deepcopy
import math


# 2.2 主类
class Combine(object):

    def __init__(self, arr_list=None):
        """
        生成排列组合
        """
        self.all_Tree = []
        self.tree_arrange_List = []  # 排列
        self.tree_combine_List = []  # 组合
        self.tree_q = 0
        arr_list = ['0'] if arr_list is None else arr_list
        self.tree_n = len(arr_list) if type(arr_list) == list else arr_list
        self.arr_list = arr_list if type(arr_list) == list else range(self.tree_n)
        self.reverse = False

    def _combine_tree(self, root, rest, depth):
        """
        生成树函数
        """
        depth += 1
        if depth <= self.tree_q:
            for each in rest:
                next_rest = deepcopy(rest)
                next_root = root + [each]
                next_rest.remove(each)
                self._combine_tree(next_root, next_rest, depth)
        else:
            root = sorted(root)
            if root not in self.tree_combine_List:
                self.tree_combine_List.append(root)

    def cnm(self, m=0):
        """
        C N 取 M 个的组合
        """
        if m > math.floor(len(self.arr_list) / 2):
            m = len(self.arr_list) - m
            self.reverse = True
        self.tree_q = m
        self._combine_tree([], self.arr_list, 0)
        if self.reverse:
            self.tree_combine_List = map(lambda x: list(set(self.arr_list).difference(set(x))), self.tree_combine_List)
        return self.tree_combine_List

    def create_tree(self):
        """
        生成组合
        """
        for each in range(1, self.tree_n + 1):
            self.tree_q = each
            self._combine_tree([], self.arr_list, 0)
        return self.tree_combine_List


# 2.3 demo运行函数
def combine_demo():
    combine = Combine(6)
    combine.cnm(2)
    print combine.tree_combine_List
    combine = Combine(list('abcdef'))
    combine.cnm(2)
    print combine.tree_combine_List


# ---------------------------------------------------------------------------
# 3、numpy的方便处理的包
# 3.1 引入包
import numpy as np


# 3.2 主类
class npl(object):
    """numpy local method : npl"""

    @staticmethod
    def loc(np_array, cols):
        """按照cols的布尔序列选出array的某些行

        :param np_array:
        :param cols: [True, False] 组成的 list 或者 np.array
        :return:

        Example
        ----------
        >>> npl.loc(test, test[:,-1] == 1)  # 抽出test最后一列为1的np.array
        """
        return np_array[np.where(cols)[0], :]

    @staticmethod
    def sort(np_array, cols, ascending=[]):
        """按cols从小到大排列array的行数据

        cols 为 [3, 1] 数字组成的 list
        使用方法如下：
        sort(test, [3, 1])           # 按照test的第4列，第2列从小到大排序
        """
        if ascending == []:
            sort_data = np_array[:, cols[::-1]].T
        else:
            sort_data = ([map(lambda x: 1 if x else -1, ascending[::-1])] * np_array[:, cols[::-1]]).T
        return np_array[np.lexsort(sort_data), :]

    @staticmethod
    def drop_duplicates(np_array):
        """
        去重
        使用方法如下：
        drop_duplicates(test)           # 去除test的重复的行
        """
        return np.array(list(set([tuple(x) for x in np_array])))


# 3.3 demo运行函数
def npl_demo():
    def get_data():
        index_data = range(8)
        sales = np.random.rand(8) * 8
        std = np.random.rand(8) * 2
        labels = [0, 1, 0, 1, 2, 0, 2, 1]
        data = np.array(zip(index_data, sales, std, labels))
        return data

    data = get_data()
    npl.loc(data, data[:, -1] == 1)  # 取出最后一列为1的行
    npl.sort(data, [1, 2], ascending=[True, False])  # 按照第0列升序，第1列降序
    npl.drop_duplicates(data)  # data数据去重，以每行数据为粒度


# ---------------------------------------------------------------------------
# 4、pandas的方便处理的包
# 4.1 引入包
import pandas as pd


# 4.2 主类
class pdl(object):

    @staticmethod
    def date_range(start_date, end_date, freq='D'):
        """返回日期函数。

        :param start_date: '2017-10-01' 开始日期
        :param end_date: '2017-11-01' 结束日期
        :param freq: 间隔频率
        :return: 日期 list

        ex:
        getDateRange(start_date, end_date, freq='D')
        getDateRange(start_date, end_date, freq='M')
        getDateRange(start_date, end_date, freq='H')
        """
        date_range = map(lambda x: str(x)[:10], pd.date_range(start_date, end_date, freq=freq).values)
        return date_range

    @staticmethod
    def trans2Str(data, cols):
        """
        讲num类型转化成str类型
        """
        for col in cols:
            data[col] = data.loc[:, [col]].applymap(str)
        return data

    @staticmethod
    def trans2Num(data, cols):
        """
        讲str类型转化成num类型
        """
        for col in cols:
            data[col] = data.loc[:, [col]].applymap(float)
        return data

    @staticmethod
    def trans2Int(data, cols):
        """
        讲str类型转化成int类型
        """
        for col in cols:
            data[col] = data.loc[:, [col]].applymap(int)
        return data

    @staticmethod
    def crossJoin(pda, pdb):
        pda['tmp_cross_join'] = '1'
        pdb['tmp_cross_join'] = '1'
        pdc = pda.merge(pdb, on=['tmp_cross_join'])
        pdc = pdc.drop(['tmp_cross_join'], axis=1)
        return pdc


# ---------------------------------------------------------------------------
# 5、logger包
# 5.1 引入包
import logging


# 5.2 主类
class Logger(object):
    def __init__(self, logger=''):
        self.getOrCreate(logger)

    @staticmethod
    def _logger(mon_str):
        def wrapper_func(func):
            def wrapper_args(self, *args, **kwargs):
                self.logger.info('{0} ...'.format(mon_str))
                result = func(self, *args, **kwargs)
                self.logger.info('{0} Finish !'.format(mon_str))
                return result

            return wrapper_args

        return wrapper_func

    def _getLogger(self, logger):
        """取得log日志 """
        self.logger = logger

    def _setLogger(self):
        """设置log日志 """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(filename)s) [line:%(lineno)d] [ %(levelname)s ] %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            # filename='ModelSolve.log',
                            # filemode='w'
                            )
        self.logger = logging.getLogger("ModelClassify")

    def getOrCreate(self, logger=''):
        if logger == '':
            self._setLogger()
        else:
            self._getLogger(logger=logger)


# ---------------------------------------------------------------------------
# 6、平滑函数
# 6.1 引入包
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings

warnings.filterwarnings('ignore')
lowess = sm.nonparametric.lowess


# 6.2 主类
class SmoothMethod(object):
    @staticmethod
    def Lowess(x, y, theta=3, frac=0.3, it=2, int_method=''):
        """Lowess Smooth

        :param x: list | np.array | pd.Series
        :param y: list | np.array | pd.Series
        :param theta: x times of error
        :param frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
        :param it: The number of residual-based reweightings to perform.
        :param int_method: int methods '' | 'ceil' | 'floor' | 'int'
        :return: np.array
        """
        int_methods = {'ceil': np.ceil, 'floor': np.floor, 'int': np.int32}
        x = np.array(x)
        y = np.array(y)
        lowes = lowess(y, x, frac=frac, it=it, return_sorted=False)
        err = y - lowes
        std_err = np.std(err)
        choose = (np.abs(err) < theta * std_err)
        if int_method == '':
            result = np.where(choose, y, lowes)
        elif int_method in ['ceil', 'floor', 'int']:
            func = int_methods[int_method]
            result = np.where(choose, func(y), func(lowes))
        else:
            raise ValueError('''int_method Must Be One Of : '' | 'ceil' | 'floor' | 'int' ! ''')
        return result

    @staticmethod
    def EMSmooth(x, y, theta=2, threshold=10, int_method=''):
        """EM Smooth

        :param x: list | np.array | pd.Series
        :param y: list | np.array | pd.Series
        :param theta: x times of error
        :param threshold: only higher than this can be thought as abnormal value
        :param int_method: int methods '' | 'ceil' | 'floor' | 'int'
        :return: np.array
        """
        int_methods = {'ceil': np.ceil, 'floor': np.floor, 'int': np.int32}
        y = np.array(y, dtype=np.float64)
        y_sort = np.argsort(np.argsort(y)[::-1])
        data = np.array([x, y, y_sort, range(len(x))]).T
        data = npl.sort(data, [2])
        flag = True
        i = 0
        while flag:
            avg = np.nanmean(data[(i + 1):, 1])
            sigma = np.nanstd(data[(i + 1):, 1])
            if (data[i, 1] <= avg + theta * sigma) | (data[i, 1] <= threshold):
                flag = False
            data[i, 1] = np.nan
            i += 1
        data = npl.sort(data, [0])
        result = pd.Series(data[:, 1]).interpolate(limit=len(data) - 1, limit_direction='both').values[
            np.argsort(data[:, -1])]
        if int_method in ['ceil', 'floor', 'int']:
            func = int_methods[int_method]
            result = func(y)
        elif int_method != '':
            raise ValueError('''int_method Must Be One Of : '' | 'ceil' | 'floor' | 'int' ! ''')
        return result

    @staticmethod
    def smooth(df, group_columns, target_columns, smooth_method='lowess', theta=-1.0, frac=-1.0, it=-1.0,
               threshold=-1.0, int_method=''):
        """DataFrame Smooth Method include lowess and emsmooth

        :param df: DataFrame
        :param group_columns: column may contain different values which should treat individually
        :param target_columns: column containing values needed to be smoothed
        :param smooth_method:
        :param theta: x times of error
        :param frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
        :param it: The number of residual-based reweightings to perform.
        :param threshold: only higher than this can be thought as abnormal value
        :param int_method: int methods '' | 'ceil' | 'floor' | 'int'
        :return: DataFrame
        """
        kwargs_map = {'lowess': [SmoothMethod.Lowess, {'theta': 3, 'frac': 0.3, 'it': 2, 'int_method': ''}],
                      'emsmooth': [SmoothMethod.EMSmooth, {'theta': 2, 'threshold': 10, 'int_method': ''}]}
        smooth_method = smooth_method.lower()
        if smooth_method in ['lowess', 'emsmooth']:
            func = kwargs_map[smooth_method][0]
            kwargs = kwargs_map[smooth_method][1]
            for each in kwargs.keys():
                tmp_arg = eval(each)
                kwargs[each] = tmp_arg if tmp_arg != -1 else kwargs[each]
        else:
            raise ValueError(''' smooth_method Must Be One Of : 'lowess' | 'emsmooth' ! ''')
        split_data = []
        for key, grouped in df.groupby(group_columns):
            for target in target_columns:
                grouped['smoothed_{0}'.format(target)] = func(range(len(grouped)), grouped[target], **kwargs)
            split_data.append(grouped)
        result = pd.concat(split_data, ignore_index=True)
        return result


# 3.3 demo运行函数
def smooth_demo():
    def get_data():
        x = np.sort(np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=500))
        y = []
        for i in xrange(8):
            y.append(np.sin(x) + np.random.normal(size=len(x)))
        y = np.concatenate(y)
        index_y = np.array(reduce(lambda m, n: m + n, [['data_{0}'.format(i)] * len(x) for i in range(8)]))
        x = np.concatenate([x] * 8)
        data = pd.DataFrame(np.matrix([x, index_y, y]).T, columns=['dt', 'dc', 'sales'])
        tmp_len = len(data)
        data = pd.concat([data] * 10)
        data['sku'] = reduce(lambda m, n: m + n, [['sku_{0}'.format(i)] * tmp_len for i in range(10)])
        data = pdl.trans2Num(data, ['dt', 'sales'])
        return data

    data = get_data()
    data_smooth3 = SmoothMethod.smooth(data, ['dc', 'sku'], ['dt', 'sales'], smooth_method='lowess')
    data_smooth3_1 = SmoothMethod.smooth(data, ['dc', 'sku'], ['dt', 'sales'], smooth_method='lowess', theta=3,
                                         frac=0.3, it=2)
    data_smooth4 = SmoothMethod.smooth(data, ['dc', 'sku'], ['dt', 'sales'], smooth_method='emsmooth')
    data_smooth4_1 = SmoothMethod.smooth(data, ['dc', 'sku'], ['dt', 'sales'], smooth_method='emsmooth', theta=2,
                                         threshold=10)
