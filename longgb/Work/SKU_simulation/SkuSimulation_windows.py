# coding=utf-8

import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import rv_discrete
from matplotlib.backends.backend_pdf import PdfPages


from StatisUtil import EMsmooth


class Utils:

    def __init__():
        pass

    @staticmethod
    def generate_date_range(start_dt, end_dt):
        if type(start_dt) == str:
            start_dt = datetime.datetime.strptime(start_dt, '%Y-%m-%d')
        if type(end_dt) == str:
            end_dt = datetime.datetime.strptime(end_dt, '%Y-%m-%d')
        step = datetime.timedelta(days=1)
        date_range = []
        while start_dt <= end_dt:
            date_range.append(start_dt.strftime('%Y-%m-%d'))
            start_dt += step
        return date_range

    @staticmethod
    def truncate_vlt_distribution(val, prob):
        vlt_mean = (val * prob).sum()
        val_trunc = np.copy(val[val >= vlt_mean])
        prob_trunc = np.copy(prob[val >= vlt_mean])
        prob_trunc[0] += np.sum(prob[~(val >= vlt_mean)])
        return val_trunc, prob_trunc

    @staticmethod
    def getCategory(sales,):
        category=99 #normal
        percent = (sum(sales>0)*1.0/ len(sales))
        salesMean = np.mean(sales)
        if  (percent>=category_longTail_stable_DaysThreshold) & (salesMean<=category_longTail_stable_SalesThreshold):
            category=1 #longTail_stable
        return category

    @staticmethod
    def getPredictionErrorMultiple(sales,pred_sales,cur_index):
        """
        judge whether prediction sales exceed the actual sales
        """
        sales3days =  sum([sales[cur_index]]*3)
        pred_sales3days = sum([pred_sales[cur_index][0]]*3)
        if cur_index >= 3:
            sales3days = sum(sales[cur_index-3:cur_index])
            pred_sales3days = sum(pred_sales[cur_index-3][0:3])
        multiple = max((sales3days*1.0/pred_sales3days),1)
        return multiple

    @staticmethod
    def getWeightedActSales(sales,cur_index):
        """
        1. estimate whether error is too large
        2. return weighted
        """
        if cur_index>= salespredictionErrorFilldays:
            actualSale = sales[cur_index-salespredictionErrorFilldays:cur_index]
            return [np.mean(actualSale)],[np.std(actualSale)]
        else:
            rang = salespredictionErrorFilldays - cur_index
            mean_sale = np.nanmean(sales[0:cur_index])
            actualSale = np.concatenate((sales[0:cur_index],np.array([mean_sale]*(rang))))
            return [np.mean(actualSale)],[np.std(actualSale)]


# =========================== 转移的字段 ===========================
sales_his = 1
inv_his = 1
sales_pred_mean = 1
vlt_val = 1
vlt_prob = 1
s=0
S=0
sku_id=''
sku_name=''
ito_level=-99
sales_per = 1
cv_sale=0

# =========================== 重新定义字段 ===========================
sale_per = sales_per
date_range = ['2016-12-02', '2016-12-15']
s_array=[]
S_array=[]
pred_days=28
org_nation_sale_num_band=None
start_dt = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')        # '2016-12-02'
end_dt = datetime.datetime.strptime(date_range[1], '%Y-%m-%d')          # '2016-12-15'
simulation_period = (end_dt - start_dt).days + 1                        # 14
sales_his_origin = sales_his                                            # 销量
inv_his = inv_his                                                       # 库存
org_nation_sale_num_band = org_nation_sale_num_band                     #
sales_his = sales_his_origin.copy()
vlt_val = vlt_val.astype(np.int32)
vlt_distribution = rv_discrete(values=(vlt_val, vlt_prob))              # vlt 的分布
latest_arrive_index = 0                                                 # 最晚的到货日期
# 处理销量预测为空的情况：使用前一天的预测值
for i in range(simulation_period):
    try:
        if sales_pred_mean[i] is None:
            sales_pred_mean[i] = sales_pred_mean[i - 1]
    except:
        print sales_pred_mean
        print sku_id
# -------------------------- 仿真数据 --------------------------
inv_sim = np.array([0] * simulation_period, dtype=np.float64)           # 库存，初始化  inv
inv_sim[0] = inv_his[0]                                                 #       第一天为历史库存
sales_sim = np.array([0] * simulation_period, dtype=np.float64)         # 销量      sales
lop = np.array([0] * simulation_period, dtype=np.float64)               # 补货点    lop
open_po_sim = np.array([0] * simulation_period, dtype=np.float64)       # 在途      openpo
arrive_qtty_sim = np.array([0] * simulation_period, dtype=np.float64)   # 到达      arrive
arrive_pur_sim = np.array([0] * simulation_period, dtype=np.float64)    # 到货批次  pur
pur_qtty_sim = [None] * simulation_period                               # 采购      pur_qtty
vlt_sim = [None] * simulation_period                                    # VLT
pur_cnt_sim = 0                                                         # 仿真采购次数    pur_cnt
success_cnt = 0                                                         # 采购成功次数    success_cnt
cur_index = 0                                                           # 初始化日期下标  cur_index
simulation_status = 0                                                   # 仿真状态：{0: 未开始，1: 进行中，2：正常完成}
subCategory = np.array([99]*simulation_period)


def run_simulation(cr=None, bp=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    simulation_status = 1
    # 1.0 遍历日期
    while cur_index < simulation_period:
        # 如果有采购到货，将采购到货加到当天的现货库存中
        # 判断本次采购是否成功：当天0点库存大于0为成功
        if arrive_qtty_sim[cur_index] > 0:
            if inv_sim[cur_index] > 0:          # 【为什么这个不大于0，就不算成功购买了。在途也满足。？】
                success_cnt += arrive_pur_sim[cur_index]
            inv_sim[cur_index] += arrive_qtty_sim[cur_index]            # 【这个地方有问题，只更新了在途，到达没有更新。】
        # 1.1 补货点
        calc_lop(cr)
        # 1.2 补货量。判断：现货库存 + 采购在途 < 补货点
        flag = (lop[cur_index] - inv_sim[cur_index] - open_po_sim[cur_index]) > 0
        if flag:
            replenishment(bp)
        # 1.3 计算销量。当天仿真销量 = min(当天历史销量, 当天仿真库存量)
        sales_sim[cur_index] = min(sales_his[cur_index], inv_sim[cur_index]+open_po_sim[cur_index])
        # 下一天的仿真库存量（初始库存） = 当天仿真库存量 - 当天销量,销量可能超过库存因为消耗在途
        # 将当天的在途进行运算，当天在途数量默认消耗不入库
        if cur_index < simulation_period - 1:
            inv_sim[cur_index + 1] = max(inv_sim[cur_index] -sales_sim[cur_index],0)
            open_po_sim[cur_index] = open_po_sim[cur_index]-max(0,sales_sim[cur_index]-inv_sim[cur_index])
        # 更新日期下标
        cur_index += 1
    simulation_status = 2


def reset():
    inv_sim = np.array([0] * simulation_period, dtype=np.float64)
    inv_sim[0] = inv_his[0]
    sales_sim = np.array([0] * simulation_period, dtype=np.float64)
    lop = [0] * simulation_period
    open_po_sim = np.array([0] * simulation_period, dtype=np.float64)
    arrive_qtty_sim = np.array([0] * simulation_period, dtype=np.float64)
    arrive_pur_sim = np.array([0] * simulation_period, dtype=np.float64)
    pur_qtty_sim = [None] * simulation_period
    vlt_sim = [None] * simulation_period
    pur_cnt_sim = 0
    success_cnt = 0
    cur_index = 0
    simulation_status = 0
    latest_arrive_index = 0

def calc_lop(cr=None):
    demand_mean = np.mean(sales_pred_mean[cur_index])
    if len(s_array)>1:
        lop[cur_index] =s_array[cur_index]*demand_mean
    else:
        lop[cur_index] =s*demand_mean

def calc_replenishment_quantity(bp=None, cr=None):
    if len(S_array)>1:
        return np.ceil(S_array[cur_index]* np.mean(sales_pred_mean[cur_index]) - inv_sim[cur_index] - open_po_sim[cur_index])
    else:
        return np.ceil(S* np.mean(sales_pred_mean[cur_index]) - inv_sim[cur_index] - open_po_sim[cur_index])

def replenishment(bp=None):
    # 计算补货量
    pur_qtty_sim[cur_index] = calc_replenishment_quantity(bp)
    # 根据VLT分布生成到货时长
    vlt_sim[cur_index] = vlt_distribution.rvs()
    # 更新补货状态，last_index是采购到货天的下标
    last_index = cur_index + vlt_sim[cur_index] + 1
    # 判断上次的到货时间点早于该次的到货时间，如果晚于则重新抽样确定vlt
    while last_index < latest_arrive_index:
        vlt_sim[cur_index] = vlt_distribution.rvs()
        last_index = cur_index + vlt_sim[cur_index] + 1
    if last_index < simulation_period:
        # 更新采购在途
        open_po_sim[(cur_index+1):last_index] += pur_qtty_sim[cur_index]
        # 更新采购到货
        arrive_qtty_sim[last_index] += pur_qtty_sim[cur_index]
        # 更新到货批次
        arrive_pur_sim[last_index] += 1
        # 更新采购次数
        pur_cnt_sim += 1
    else:
        # 只更新采购在途
        open_po_sim[(cur_index+1):] += pur_qtty_sim[cur_index]
    # 更新next_arrive_index
    latest_arrive_index = last_index

def get_daily_data():
    if simulation_status == 2:
        if len(s_array)>1:
            daily_data = {'dt': Utils.generate_date_range(start_dt, end_dt),
                          'sales_his_origin': sales_his_origin,
                          'inv_his': inv_his,
                          'sales_sim': sales_sim,
                          'inv_sim': inv_sim,
                          'lop': lop,
                          'pur_qtty_sim': pur_qtty_sim,
                          'open_po_sim': open_po_sim,
                          'vlt_sim': vlt_sim,
                          'arrive_qtty_sim': arrive_qtty_sim,
                          'sales_pred_mean': sales_pred_mean,
                          'sku_id': np.repeat(sku_id, simulation_period),
                          'sale_num_band':org_nation_sale_num_band,
                          'subCategory' : subCategory,
                          's':s_array,
                          'S':S_array}
            return pd.DataFrame(daily_data, columns=['sku_id', 'dt', 'sales_his_origin', 'inv_his', 'sales_sim',
                                                 'inv_sim', 'lop', 'pur_qtty_sim', 'open_po_sim',
                                                 'vlt_sim', 'arrive_qtty_sim', 'sales_pred_mean','sale_num_band','subCategory','s','S'])
        else:
            daily_data = {'dt': Utils.generate_date_range(start_dt, end_dt),
                      'sales_his_origin': sales_his_origin,
                      'inv_his': inv_his,
                      'sales_sim': sales_sim,
                      'inv_sim': inv_sim,
                      'lop': lop,
                      'pur_qtty_sim': pur_qtty_sim,
                      'open_po_sim': open_po_sim,
                      'vlt_sim': vlt_sim,
                      'arrive_qtty_sim': arrive_qtty_sim,
                      'sales_pred_mean': sales_pred_mean,
                      'sku_id': np.repeat(sku_id, simulation_period),
                      'sale_num_band':org_nation_sale_num_band,
                      'subCategory' : subCategory}
            return pd.DataFrame(daily_data, columns=['sku_id', 'dt', 'sales_his_origin', 'inv_his', 'sales_sim',
                                                  'inv_sim', 'lop', 'pur_qtty_sim', 'open_po_sim',
                                                 'vlt_sim', 'arrive_qtty_sim', 'sales_pred_mean','sale_num_band','subCategory'])
    else:
        print "Please call run_simulation() before getting daily data!"

def calc_kpi():
    if simulation_status == 2:
        # 现货率（cr）：有货天数除以总天数
        cr_sim = (inv_sim > 0).sum() / float(simulation_period)
        cr_his = (inv_his > 0).sum() / float(simulation_period)
        # 周转天数（ito）：平均库存除以平均销量
        ito_sim = np.nanmean(inv_sim) / np.nanmean(sales_sim)
        ito_his = np.nanmean(inv_his) / np.nanmean(sales_his_origin)
        # 总销量（ts）
        ts_sim = np.sum(sales_sim)
        ts_his = np.sum(sales_his_origin)
        ts_rate=ts_sim/ts_his
        if sale_per is None:
            return [sku_id, round(cr_sim,4), round(cr_his,4), round(ito_sim,4), round(ito_his,4), round(ts_sim,4), round(ts_his,4),
                    pur_cnt_sim,s,S,date_range[0],date_range[1],ito_level,ts_rate,-1,cv_sale]
        else:
            sku_kpi = [sku_id, round(cr_sim, 4), round(cr_his, 4), round(ito_sim, 4), round(ito_his, 4), round(ts_sim, 4),
             round(ts_his, 4), pur_cnt_sim, s, S, date_range[0], date_range[1], ito_level, ts_rate, sale_per[0], cv_sale]
            return [sku_id, round(cr_sim,4), round(cr_his,4), round(ito_sim,4), round(ito_his,4), round(ts_sim,4), round(ts_his,4),
                    pur_cnt_sim,s,S,date_range[0],date_range[1],ito_level,ts_rate,sale_per[0],cv_sale]
    else:
        print "Please call run_simulation() before calculating KPI!"

