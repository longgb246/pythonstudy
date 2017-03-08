#-*- coding:utf-8 -*-
from __future__ import division
import os
import pandas as pd
import time
import numpy as np
from collections import Counter

# ======================== 功能函数 ========================
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
        print 'Run Time{3} is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print 'Run Time{2} is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


def getDateRange(start_date, end_date, freq='D'):
    date_range = map(lambda x: str(x)[:10], pd.date_range(start_date, end_date, freq=freq).values)
    return date_range


# ======================== 实现函数 ========================
def loadData(read_path):
    '''
    读取数据
    '''
    load_file = read_path + os.sep + 'tmp_oneday_rdc_info_combine.out'
    # 读取时间 Run Time is : 0.0 min 33.0780 s
    t1 = time.time()
    oneday_data = pd.read_table(load_file, header=None)
    printRunTime(t1, 'Read Data')
    columns = ['rdc_id', 'sku_id', 'item_second_cate_cd', 'item_third_cate_cd', 'open_po', 'inv', 'variance', 'ofdsales', 'order_date', 'total_sales', 'date_s']
    oneday_data.columns = columns
    return oneday_data


def countData(oneday_data, columns=[]):
    '''
    按指定列统计缺失的数据
    '''
    if columns==[]:
        columns = oneday_data.columns.tolist()
    len_data = len(oneday_data)
    count_data = {}
    # t1 = time.time()
    for each in columns:
        # each = columns[0]
        tmp_count = dict(Counter(oneday_data[each].isnull()))           # 空数据
        try:
            count_data[each] = (len_data - tmp_count[False])/ len_data  # 缺失率
        except:
            try:
                count_data[each] = tmp_count[True] / len_data           # 缺失率
            except:
                count_data[each] = np.nan
    # printRunTime(t1, 'Count Data')
    return count_data


def countDataByDate_s(oneday_data_0219):
    '''
    按日期统计缺失的数据
    '''
    t1 = time.time()
    all_count_bydate = pd.DataFrame()
    for date_s in date_range:
        count_oneday_data_0219_2 = countData(oneday_data_0219[oneday_data_0219['date_s'] == date_s], columns=['ofdsales', 'order_date', 'total_sales', 'variance'])
        count_oneday_data_0219_2_tmp = pd.DataFrame.from_dict(count_oneday_data_0219_2, orient='index').T
        count_oneday_data_0219_2_tmp['date_s'] = date_s
        all_count_bydate = pd.concat([all_count_bydate, count_oneday_data_0219_2_tmp])
    all_count_bydate.index = range(len(all_count_bydate))
    printRunTime(t1, 'count by date_s')
    # all_count_bydate = all_count
    return all_count_bydate


# ======================== 配置参数 ========================
read_path = r'D:\Lgb\data_sz'
save_path = r'D:\Lgb\WorkFiles\One_day'
start_date = '2016-02-19'
end_date = '2017-02-27'
date_range = getDateRange(start_date, end_date)


if __name__ == '__main__':
    oneday_data = loadData(read_path)
    oneday_data_0219 = oneday_data[(oneday_data['date_s'] >= '2016-02-19') & (oneday_data['date_s'] <= '2017-02-27')]
    count_oneday_data = countData(oneday_data)
    count_oneday_data_0219 = countData(oneday_data_0219)
    # 发现现在的缺失发生在 'ofdsales', 'total_sales', 'variance'
    # 'order_date' 这个缺失，不关心。
    # 主要关注 count_oneday_data_0219 这个数据集
    all_count_bydate = countDataByDate_s(oneday_data_0219)
    # all_count_bydate[all_count_bydate['ofdsales'] == 1]       和 variance 一样,
    # all_count_bydate[all_count_bydate['total_sales'] == 1]    是空的
    # all_count_bydate[all_count_bydate['variance'] == 1]       和 ofdsales 一样
    # print all_count_bydate[all_count_bydate['ofdsales'] == 1]['date_s'].tolist()
    # 发现这些天的数据都没有销量预测的数据。
    missing_list = ['2016-03-28', '2016-03-29', '2016-03-30', '2016-04-07', '2016-04-15', '2016-04-16', '2016-04-17', '2016-04-30',
     '2016-05-13', '2016-05-26', '2016-06-14', '2016-07-03', '2016-07-16', '2016-07-18', '2016-08-14', '2016-09-14',
     '2016-09-21', '2016-09-24', '2016-09-25', '2016-09-27', '2016-10-17', '2016-10-26', '2016-10-28', '2016-11-24',
     '2017-02-13']
    for each in missing_list:
        try:
            date_range.remove(each)
        except:
            pass
    oneday_data_0219_no_missinglist = oneday_data_0219[map(lambda x: x not in missing_list, oneday_data_0219['date_s'].values)]
    oneday_data_0219_no_missinglist_need = oneday_data_0219_no_missinglist.loc[:, ['sku_id', 'rdc_id', 'ofdsales', 'total_sales', 'variance', 'date_s']]
    oneday_data_0219_no_missinglist_need['ofdsales_mask'] = map(lambda x: 1 if x else 0,oneday_data_0219_no_missinglist_need['ofdsales'].notnull().values)
    oneday_data_0219_no_missinglist_need['total_sales_mask'] = map(lambda x: 1 if x else 0,oneday_data_0219_no_missinglist_need['total_sales'].notnull().values)
    oneday_data_0219_no_missinglist_need['variance_mask'] = map(lambda x: 1 if x else 0,oneday_data_0219_no_missinglist_need['variance'].notnull().values)
    sku_list = oneday_data_0219['sku_id'].drop_duplicates().tolist()
    # 分情况讨论
    # 1、有数据 -> 没数据 -> 有数据
    # 2、有数据 -> 没数据
    # 3、没数据 -> 有数据
    # 4、没数据 -> 有数据 -> 没数据
    oneday_data_0219_situation = pd.DataFrame()
    for sku_id in sku_list:
        sku_id = sku_list[0]
        for date_this in date_range:
            date_this = date_range[0]
            oneday_data_0219_no_missinglist_need[(oneday_data_0219_no_missinglist_need['sku_id'] == sku_id) & (oneday_data_0219_no_missinglist_need['date_s'] == date_this)]
            oneday_data_0219_situation = pd.concat([oneday_data_0219_situation, ])
    pass

