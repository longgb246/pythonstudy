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


if __name__ == '__main__':
    printRunTimeDemo()
    pass
