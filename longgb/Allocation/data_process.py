# coding=utf-8
from sys import path
import os
import sys

pth = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
# 当前包路径导入
# __file__ = r'D:\Lgb\pythonstudy\longgb\analysis\analysis_run.py'
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Allocation.inventory_process import inventory_proess
import pandas as pd
import datetime
from collections import defaultdict
from itertools import ifilter
import numpy as np
# from com.jd.test.zhangjs.Allocation.inventory_process import inventory_proess


def datelist(start, end):
    start_date = datetime.datetime.strptime(start, '%Y%m%d')
    end_date = datetime.datetime.strptime(end, '%Y%m%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


# ====================================================================================
# =                                 （1）数据读入                                     =
# ====================================================================================
# 指定数据相关路径
sku_data_path = 'E://Allocation_data//sku_data.csv'
fdc_data_path = 'E://Allocation_data//fdc_data.csv'
order_data_path = 'E://Allocation_data//order_data.csv'
sale_data_path = 'E://Allocation_data//sale_data.csv'
# 读入相关数据集
# 读入SKU粒度数据集
'''date,sku_id,dc_id,forecast_begin_date,forecast_days,forecast_daily_override_sales,
forecast_weekly_override_sales,forecast_weekly_std,inv,arrive_quantity,open_po,white_flag'''
allocation_sku_data = pd.read_csv(sku_data_path, date_parser='date')
# 读入FDC粒度数据
'''
date,dc_id,alt,alt_prob
'''
allocation_fdc_data = pd.read_csv(fdc_data_path, date_parser='date')
# 读入采购单粒度数据
'''
date,rdc,order_id,item_sku_id,arrive_quantity
'''
allocation_order_data = pd.read_csv(order_data_path, date_parser='date')
# 读入订单明细数据
'''
date,sale_ord_id,item_sku_id,sale_qtty,sale_ord_tm,sale_ord_type,sale_ord_white_flag
'''
allocation_sale_data = pd.read_csv(sale_data_path, date_parser='date')


# 定义索引转换函数
def gene_index(self, fdc, sku, date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return date_s + fdc + sku


##将上述读入的数据集，转换为调拨仿真类需要的数据集
##标记仿真的开始和结束日期
start_date = '2016/10/22'
end_date = '20161201'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# （1）.astype()
# a = pd.DataFrame(np.matrix([['1','2','3'],['2','4','3']]), columns=list("ABC"))
# a['A'].astype('int') + a['B'].astype('int')
# （2）.to_dict()
# b = a.set_index('A')['B'].to_dict()
# （3）defaultdict()
# white_list_dict = defaultdict(lambda: defaultdict(list))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 预测数据相关信息{fdc_sku_date:[7 days sales]},{fdc_sku_data:[7 days cv]}
fdc_forecast_sales = pd.concat([allocation_sku_data['date'].astype('str') + allocation_sku_data['dc_id'].astype('str')
                                + allocation_sku_data['sku_id'].astype('str'),
                                allocation_sku_data['forecast_daily_override_sales']], axis=1)
fdc_forecast_sales.columns = ['id', 'forecast_value']
fdc_forecast_sales = fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()

fdc_forecast_std = pd.concat([allocation_sku_data['date'].astype('str') + allocation_sku_data['dc_id'].astype('str')
                              + allocation_sku_data['sku_id'].astype('str'),
                              allocation_sku_data['forecast_weekly_std']], axis=1)
fdc_forecast_std.columns = ['id', 'forecast_std']
fdc_forecast_std = fdc_forecast_std.set_index('id')['forecast_std'].to_dict()
# *********【【【【上面是每日的 forecast_daily_override_sales ，下面是每周的 forecast_weekly_std 】】】】*********

# RDC-->FDC时长分布,{fdc:[days]}}
fdc_alt = pd.concat([allocation_fdc_data['date'].astype('str') + allocation_fdc_data['dc_id'].astype('str'),
                     allocation_fdc_data['alt']], axis=1)
fdc_alt.columns = ['id', 'alt']
fdc_alt = fdc_alt.set_index('id')['alt'].to_dict()

fdc_alt_prob = pd.concat([allocation_fdc_data['date'].astype('str') + allocation_fdc_data['dc_id'].astype('str'),
                          allocation_fdc_data['alt_prob']], axis=1)
fdc_alt_prob.columns = ['id', 'alt_prob']
fdc_alt_prob = fdc_alt_prob.set_index('id')['alt_prob'].to_dict()

# defaultdict(lamda:defaultdict(int)),只需要一个初始化库存即可
# [龙]：取开始日期的数据
tmp_df = allocation_sku_data[allocation_sku_data['date'] == start_date]
fdc_inv = pd.concat([tmp_df['date'].astype(str) + tmp_df['dc_id'].astype(str) + tmp_df['sku_id'].astype(str),
                     tmp_df['inv']], axis=1).set_index(0)['inv'].to_dict()

# 白名单,不同日期的白名单不同{fdc:{date_s:[]}}
white_list_dict = defaultdict(lambda: defaultdict(list))
tmp_df = allocation_sku_data[allocation_sku_data['white_flag'] == 1][['date', 'sku_id', 'dc_id']]
for k, v in tmp_df['sku_id'].groupby([tmp_df['date'], tmp_df['dc_id']]):
    white_list_dict[k[0]][k[1]] = list(v)
# 调拨量字典,fdc_allocation=defaultdict(float)
fdc_allocation = ''
# fdc列表：
fdc = pd.unique(allocation_fdc_data['dc_id'])
# RDC库存，{date_sku_rdc:库存量} defaultdict(int),只需要初始化的RDC库存
rdc_inv = defaultdict(int)
rdc_inv.update({i[0]: i[1] for i in fdc_inv.items() if 'rdc' in i[0]})
# 采购单数据，采购ID，SKU，实际到达量，到达时间,将其转换为{到达时间:{SKU：到达量}}形式的字典，defaultdict(lambda :defaultdict(int))
tmp_df = allocation_order_data[['date', 'item_sku_id', 'arrive_quantity']]
order_list = defaultdict(lambda: defaultdict(int))
order_list.update(tmp_df.set_index(['date', 'item_sku_id']).unstack(0)['arrive_quantity'].to_dict())

# 仿真的时间窗口 时间格式如下：20161129,这个怎么使用日期区间，待定
date_range = datelist('20160101', end_date)
# 订单数据：{订单时间_订单id:{SKU：数量}},当前的存储会造成的空间浪费应当剔除大小为0的SKU
tmp_df = allocation_sale_data[['date', 'item_sku_id', 'sale_ord_id', 'sale_ord_tm', 'sale_qtty']]
tmp_df = tmp_df.loc[[1, 2, 3]]
tmp_df = pd.DataFrame(tmp_df)
orders_retail = pd.concat(
    [tmp_df['sale_ord_tm'].astype(str) + tmp_df['sale_ord_id'].astype(str), tmp_df[['item_sku_id', 'sale_qtty']]],
    axis=1)
orders_retail.columns = ['id', 'item_sku_id', 'sale_qtty']
orders_retail = orders_retail.set_index(['id', 'item_sku_id']).unstack(0)['sale_qtty'].to_dict()
orders_retail_new = defaultdict(lambda: defaultdict(int))
# print orders_retail_new
# print orders_retail
for k, v in orders_retail.items():
    orders_retail_new[k] = dict(filter(lambda i: np.isnan(i[1]) == False, v.items()))
print orders_retail_new
#     for k1,v1 in v.items():
#         print k1,type(k1)
#         print v1,type(v1)
# 订单类型:{订单id:类型}
orders_retail_type = defaultdict(str)
# sku当天从FDC的出库量，从RDC的出库量
sku_fdc_sales = defaultdict(int)
sku_rdc_sales = defaultdict(int)
# 全量SKU列表
all_sku_list = list(set(allocation_sku_data['sku_id']))

###初始化仿真类，并运行相关结果
allocation = inventory_proess(fdc_forecast_sales, fdc_forecast_std, fdc_alt, fdc_alt_prob, fdc_inv, white_list_dict,
                              fdc_allocation, fdc, rdc_inv,
                              order_list, date_range, orders_retail, all_sku_list)
allocation.OrdersSimulation()

#####计算KPI，KPI主要包括本地订单满足率，周转，SKU满足率
# 本地订单满足率 (本地出库订单+订单驱动内配)/订单数量
cnt_orders_retail_type = {}
for k, v in allocation.orders_retail_type.items():
    cnt_orders_retail_type.setdefault(v, []).append(k)
for k, v in cnt_orders_retail_type.items():
    print k, 'has orders number:', len(v)
    # 周转，考核单个SKU的周转
    # print allocation.fdc_inv
