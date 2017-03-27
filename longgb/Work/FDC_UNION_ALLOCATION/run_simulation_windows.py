# coding: utf-8
import os
from sys import path
pth = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
import ast
import datetime
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import time
from collections import defaultdict
import csv
import dill
import copy
import sys

from StatisUtil import EMsmooth
from inventory_process_online import inventory_proess


def datelist(start, end):
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


def fill_nan(input_array):
    if isinstance(input_array, np.ndarray):
        for idx in range(len(input_array)):
            if idx == 0 and np.isnan(input_array[idx]):
                input_array[idx] = np.nanmean(input_array)
            elif np.isnan(input_array[idx]):
                input_array[idx] = input_array[idx - 1]
    else:
        print 'Input must be 1d numpy.ndarray!'


def gene_index(fdc, sku, date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s) + ':' + str(fdc) + ':' + str(sku)


# =====================================================================
# =                              0、配置信息                           =
# =====================================================================
workingfolderName = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
data_dir0 = 'D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/'
output_dir = 'D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/simulation_results/'
data_file_name = 'sales_all_detail_20170324073521.csv'
rdc_sale_file_name = 'rdc_Sale_data_070203_20170324143241.csv'
order_file_name = 'order_sample_070203_20170324143010.csv'
rdc_inv_file_name = 'rdc_inv_big0_03_20170324140531.csv'
starttime = datetime.datetime.now()
date_end = '2017-03-20'
date_start = '2017-02-19'
train_date_range = ['2016-12-02','2016-12-15']
fdc_alt = {'605':[1,2],'634':[2,3],'635':[3,4]}
fdc_alt_prob = {'605':[0.9944,0.0056],'634':[0.9944,0.0056],'635':[0.9944,0.0056]}
# --------------------------- 本次仿真配置信息 ---------------------------
write_daily_data = True
write_original_data = True
train_simulation_name = 'train_kpi'
order_flag = 0
order_list_file_name = 'order_list_dict'
rdc_sale_flag = 0
rdc_sale_file_dict_name = 'rdc_sale_list_dict'
rdc_inv_flag = 0
rdc_inv_file_dict_name = 'rdc_list_inv_dict'


# ==========================================================================
# =                                1、数据处理                              =
# ==========================================================================
dtype = {'sku_id': str}
df = pd.read_csv(data_dir0 + data_file_name, sep='\t', dtype=dtype)
df.fillna(0, inplace=True)
# 对销量进行平滑
flag_smooth = 2
if flag_smooth == 1:
    df_smooth_pre = df.loc[:, ["sku_id", "dt", "total_sale"]]
    df_smooth_pre.columns = ["sku", "date", "sales"]
    df_smooth = EMsmooth.smooth(df_smooth_pre)
    df["dt"] = df["dt"].astype(str)
    df_smooth["date"] = df_smooth["date"].astype(str)
    df = pd.merge(df, df_smooth, how='inner', left_on=["sku_id", "dt"], right_on=["sku", "date"])
    df = df.loc[:,
         ["fdc_id", "sku_id", "ofdsales", "stock_qtty", "dt", "sales_smooth", "safestock", "maxstock", "total_sale_per",
          "av_sale"]]
    df.columns = ["fdc_id", "sku_id", "ofdsales", "stock_qtty", "dt", "total_sale", "safestock", "maxstock",
                  'ts_percent', 'cv_sale']
    df.fillna(0, inplace=True)  # 对销量进行平滑后，total_sale会有nan，需要填充0
    df.to_csv(path_or_buf=output_dir + 'smooth_datasets.csv', index=False, sep='\t')
elif flag_smooth == 0:
    df = pd.read_csv(output_dir + 'smooth_datasets.csv', sep='\t', dtype=dtype)
elif flag_smooth == 2:
    df = df.loc[:,
         ["dt", "fdcid", "sku_id", "ofdsales", "total_sales", "stock_qtty", "safestock", "maxstock", "stock_qtty_real",
          "varsales", "white_flag"]]
    df.columns = ["dt", "fdcid", "sku_id", "ofdsales", "total_sale", "stock_qtty_last", "safestock", "maxstock",
                  "stock_qtty", "varsales", "white_flag"]
    df.fillna(0, inplace=True)  # 对销量进行平滑后，total_sale会有nan，需要填充0
sku_list = df.sku_id.unique()
total_sku = len(sku_list)


# ------------------------------- 处理采购单数据 ------------------------------
if order_flag == 1:
    allocation_order_data = pd.read_csv(data_dir0 + order_file_name, sep='\t', dtype=dtype)
    allocation_order_data.columns = ['arrive_time', 'item_sku_id', 'arrive_quantity', 'dc_id']
    tmp_df = allocation_order_data[['arrive_time', 'item_sku_id', 'arrive_quantity']]
    tmp_df.columns = ['date', 'item_sku_id', 'arrive_quantity']
    tmp_df = tmp_df[tmp_df['date'] > date_start]
    order_list = defaultdict(lambda: defaultdict(int))
    for index, row in tmp_df.iterrows():
        if order_list.has_key(row['date']):
            if order_list[row['date']].has_key(row['item_sku_id']):
                order_list[row['date']][row['item_sku_id']] = order_list[row['date']][row['item_sku_id']] + row[
                    'arrive_quantity']
            else:
                order_list[row['date']][row['item_sku_id']] = row['arrive_quantity']
        else:
            order_list[row['date']] = {row['item_sku_id']: row['arrive_quantity']}
    filename = open(data_dir0 + order_list_file_name, 'wb')
    pickle.dump(order_list, filename)
    filename.close()
else:
    filename = open(data_dir0 + order_list_file_name, 'rb')
    order_list = pickle.load(filename)
    filename.close()


# ------------------------------- 更新增加RDC销量 -------------------------------
if rdc_sale_flag == 1:
    rdc_sale_list = defaultdict(lambda: defaultdict(int))
    allocation_rdc_sale_data = pd.read_csv(data_dir0 + rdc_sale_file_name, sep='\t', dtype=dtype)
    tmp_allocation_rdc_sale_data = allocation_rdc_sale_data.loc[:, ['sku_id', 'dt', 'total_sales']]
    tmp_allocation_rdc_sale_data = tmp_allocation_rdc_sale_data[tmp_allocation_rdc_sale_data['dt'] >= date_start]
    for index, row in tmp_allocation_rdc_sale_data.iterrows():
        if rdc_sale_list.has_key(row['dt']):
            if rdc_sale_list[row['dt']].has_key(row['sku_id']):
                rdc_sale_list[row['dt']][row['sku_id']] = rdc_sale_list[row['dt']][row['sku_id']] + row['total_sales']
            else:
                rdc_sale_list[row['dt']][row['sku_id']] = row['total_sales']
        else:
            rdc_sale_list[row['dt']] = {row['sku_id']: row['total_sales']}
    filename = open(data_dir0 + rdc_sale_file_dict_name, 'wb')
    pickle.dump(rdc_sale_list, filename)
    filename.close()
else:
    filename = open(data_dir0 + rdc_sale_file_dict_name, 'rb')
    rdc_sale_list = pickle.load(filename)
    filename.close()


# ------------------------------- RDC库存数据 -------------------------------
# RDC库存数据,获取RDC初始库存即可
if rdc_inv_flag == 1:
    rdc_inv = defaultdict(int)
    allocation_rdc_data = pd.read_csv(data_dir0 + rdc_inv_file_name, sep='\t', dtype=dtype)
    allocation_rdc_data = allocation_rdc_data[allocation_rdc_data['dt'] >= date_start]
    for kk, row in allocation_rdc_data.iterrows():
        index_rdc = gene_index('rdc', row['sku_id'], row['dt'])
        rdc_inv[index_rdc] = row['stock_qtty']
    filename = open(data_dir0 + rdc_inv_file_dict_name, 'wb')
    pickle.dump(rdc_inv, filename)
    filename.close()
else:
    filename = open(data_dir0 + rdc_inv_file_dict_name, 'rb')
    rdc_inv = pickle.load(filename)
    filename.close()


# ------------------------------- 补充其他变量 -------------------------------
fdc_list = [605, 634, 635]
mid_date_range = ['2017-02-19','2017-03-20']
date_range = datelist(mid_date_range[0], mid_date_range[1])
save_data_path = output_dir


###################################################################################################
###########################  根据仿真训练结果在测试集上进行效果测试  ################################
###################################################################################################
# ------------------------------- 保留整体明细数据 -------------------------------
system_retail_datasets = []                                     # 系统参数的明细
sim_retail_datasets = []                                        # 补货点逻辑的结果
test_date_range = ['2017-02-19','2017-03-20']
sim_fdc_sku_kpi = []                                            # 补货点逻辑的KPI
system_fdc_sku_kpi = []                                         # 系统参数的KPI
sim_lable = 'sim'                                               # 补货点逻辑的数据
system_label = 'system'                                         # 系统参数的数据

test_start_dt = datetime.datetime.strptime(test_date_range[0], '%Y-%m-%d')
test_end_dt = datetime.datetime.strptime(test_date_range[1], '%Y-%m-%d')
test_length = (test_end_dt - test_start_dt).days + 1
complete_sku = 0
sku_return_result = defaultdict(int)


for sku_id in sku_list:
    # sku_id = '1000023352'
    df_sku = df[df.sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.dt < test_date_range[0]) | (df_sku.dt > test_date_range[1])].index)
    # if (df_sku.shape[0] >= test_length) and (len(df_sku.ofdsales) > 0) and (isinstance(df_sku.ofdsales.iloc[0], str)) \
    #         and (not np.isnan(df_sku.stock_qtty.iloc[0])) and (abs(np.sum(df_sku.total_sale) - 0) > 1e-3):
    #     print sku_id    # 1000023352
    #     break
    # ---------------------------------- 不进行仿真情况 ----------------------------------
    if df_sku.shape[0] < test_length:
        continue
    if len(df_sku.ofdsales) <= 0:
        continue
    if (not isinstance(df_sku.ofdsales.iloc[0], str)):
        sku_return_result['forecastisnull'] += sku_return_result['forecastisnull']
        continue
    if np.isnan(df_sku.stock_qtty.iloc[0]):
        sku_return_result['invbeginisnull'] += sku_return_result['invbeginisnull']
        continue
    if abs(np.sum(df_sku.total_sale) - 0) <= 1e-3:
        sku_return_result['totalsalesis0'] += sku_return_result['totalsalesis0']
        continue
    sku_return_result['simsucess'] += sku_return_result['simsucess']
    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + '_origin.csv', index=False, sep='\t')
    sku_name = ''
    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + u'_origin.csv', index=False, sep='\t')
    # 保存预测数据
    fdc_forecast_sales = defaultdict(list)
    fdc_forecast_std = defaultdict(list)
    fdc_inv = defaultdict(lambda: defaultdict(int))
    white_flag = defaultdict(lambda: defaultdict(int))
    sales_retail = defaultdict(int)
    fdc_his_inv = defaultdict(int)
    system_small_s = defaultdict(int)
    system_bigger_S = defaultdict(int)
    for ky, row in df_sku.iterrows():
        # df_sku.columns
        if row['fdcid'] not in fdc_list:
            continue
        index = gene_index(row['fdcid'], row['sku_id'], row['dt'])
        # 加入预测均值和预测标准差
        if type(row['ofdsales']) == float and math.isnan(row['ofdsales']):
            fdc_forecast_sales[index].append(None)
        else:
            fdc_forecast_sales[index].extend(ast.literal_eval(row['ofdsales']))
        # 加入预测标准差
        fdc_inv[index]['inv'] = row['stock_qtty']
        sales_retail[index] = row['total_sale']
        fdc_his_inv[index] = row['stock_qtty']
        fdc_forecast_std[index] = row['varsales']
        white_flag[row['dt']][row['fdcid']] = row['white_flag']
        # 如果运行系统调拨量，需要用到s,S
        system_small_s[index] = row["safestock"]
        system_bigger_S[index] = row["maxstock"]
    # 传入的为引用，所以需要对起进行深度copy,后续system仿真使用,主要是需要进行读写操作的类型
    fdc_inv_system = copy.deepcopy(fdc_inv)
    # 按照公式计算补货点和补货量，system_flag=0
    fdc_allocation = inventory_proess(sku=sku_id, fdc_forecast_sales=fdc_forecast_sales,
                                      fdc_forecast_std=fdc_forecast_std,
                                      fdc_alt=fdc_alt, fdc_alt_prob=fdc_alt_prob, fdc_inv=fdc_inv,
                                      white_flag=white_flag,
                                      fdc_list=fdc_list, rdc_inv=rdc_inv, date_range=date_range,
                                      sales_retail=sales_retail,
                                      order_list=order_list, fdc_his_inv=fdc_his_inv, system_small_s=system_small_s,
                                      system_bigger_S=system_bigger_S, system_flag=0, rdc_sale_list=rdc_sale_list,
                                      logger=logger, save_data_path=save_data_path)
    fdc_allocation.allocationSimulation()
    # 按照SKU保存数据
    if write_daily_data:
        daily_data = fdc_allocation.get_daily_data()
        daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '_' + sim_lable + '.csv', index=False, sep='\t')
        sim_retail_datasets.append(daily_data)
    # KPI计算函数，计算出各个FDC的SKU对应的KPI
    kpi_result_sku_fdc = fdc_allocation.calc_kpi()
    kpi_result_sku_fdc.to_csv(path_or_buf=output_dir + str(sku_id) + '_' + sim_lable + '_kpi.csv', index=False,
                              sep='\t')
    sim_fdc_sku_kpi.append(kpi_result_sku_fdc)
    # 按照系统参数进行计算，system_flag=1
    fdc_allocation = inventory_proess(sku=sku_id, fdc_forecast_sales=fdc_forecast_sales,
                                      fdc_forecast_std=fdc_forecast_std,
                                      fdc_alt=fdc_alt, fdc_alt_prob=fdc_alt_prob, fdc_inv=fdc_inv_system,
                                      white_flag=white_flag,
                                      fdc_list=fdc_list, rdc_inv=rdc_inv, date_range=date_range,
                                      sales_retail=sales_retail,
                                      order_list=order_list, fdc_his_inv=fdc_his_inv, system_small_s=system_small_s,
                                      system_bigger_S=system_bigger_S, system_flag=1, rdc_sale_list=rdc_sale_list,
                                      logger=logger, save_data_path=save_data_path)
    fdc_allocation.allocationSimulation()
    # 按照SKU保存数据
    if write_daily_data:
        daily_data = fdc_allocation.get_daily_data()
        daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '_' + system_label + '.csv', index=False, sep='\t')
        system_retail_datasets.append(daily_data)
    # KPI计算函数，计算出各个FDC的SKU对应的KPI
    kpi_result_sku_fdc = fdc_allocation.calc_kpi()
    kpi_result_sku_fdc.to_csv(path_or_buf=output_dir + str(sku_id) + '_' + system_label + '_kpi.csv', index=False,
                              sep='\t')
    system_fdc_sku_kpi.append(kpi_result_sku_fdc)
    endtime = datetime.datetime.now()
    used_seconds = (endtime - starttime).total_seconds()
    complete_sku += 1
    logger.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                str(used_seconds))

# 保留补货点计算的补货参数
sim_result_kpi_df = pd.concat(sim_fdc_sku_kpi)
sim_result_kpi_df.to_csv(path_or_buf=output_dir + sim_lable + '_all_sku_kpi.csv', index=False, sep='\t')
sim_retail_df = pd.concat(sim_retail_datasets)
sim_retail_df.to_csv(path_or_buf=output_dir + sim_lable + '_all_sku_retail.csv', index=False, sep='\t')

# 保留系统的参数
system_result_kpi_df = pd.concat(system_fdc_sku_kpi)
system_result_kpi_df.to_csv(path_or_buf=output_dir + system_label + '_all_sku_kpi.csv', index=False, sep='\t')
system_retail_df = pd.concat(system_retail_datasets)
system_retail_df.to_csv(path_or_buf=output_dir + system_label + '_all_sku_retail.csv', index=False, sep='\t')
# 计算总体KPI

# 计算补货点计算的KPI
fdc_kpi = defaultdict(lambda: defaultdict(float))
for tmp_fdcid, fdcdata in sim_retail_df.groupby(['fdc_id']):
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid] = sum(fdcdata.inv_his > 0) / float(len(date_range))
        fdc_kpi['cr_sim'][tmp_fdcid] = sum(fdcdata.inv_sim > 0) / float(len(date_range))
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_sim)) <= 0 else float(
            np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin)) <= 0 else float(
            np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid] = -1 if float(fdc_kpi['ts_his'][tmp_fdcid]) <= 0 else float(
            fdc_kpi['ts_sim'][tmp_fdcid]) / float(fdc_kpi['ts_his'][tmp_fdcid])
sim_fdc_kpi = pd.DataFrame(fdc_kpi)
sim_fdc_kpi.reset_index(inplace=True)
sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
# 计算系统的KPI
fdc_kpi = defaultdict(lambda: defaultdict(float))
for tmp_fdcid, fdcdata in system_retail_df.groupby(['fdc_id']):
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid] = sum(fdcdata.inv_his > 0) / float(len(date_range))
        fdc_kpi['cr_sim'][tmp_fdcid] = sum(fdcdata.inv_sim > 0) / float(len(date_range))
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_sim)) <= 0 else float(
            np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin)) <= 0 else float(
            np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid] = -1 if float(fdc_kpi['ts_his'][tmp_fdcid]) <= 0 else float(
            fdc_kpi['ts_sim'][tmp_fdcid]) / float(fdc_kpi['ts_his'][tmp_fdcid])
system_fdc_kpi = pd.DataFrame(fdc_kpi)
system_fdc_kpi.reset_index(inplace=True)
system_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
fdc_kpi = pd.merge(sim_fdc_kpi, system_fdc_kpi, on=['fdc_id'], suffixes=['_sim', '_system'])
fdc_kpi.to_csv(path_or_buf=output_dir + 'fdc_kpi.csv', index=False, sep='\t')
print sku_return_result
