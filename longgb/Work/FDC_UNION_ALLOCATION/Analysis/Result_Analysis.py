#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


# ======================================================================
# =                                 配置参数                            =
# ======================================================================
read_path = r'D:\Lgb\WorkFiles\FDC_UNION_ALLOCATION\simulation_results'


def lagFun(array):
    lag_0 = array
    lag_1 = np.array(list(lag_0[1:]) + [0])
    lag_2 = np.array(list(lag_1[1:]) + [0])
    lag = lag_0 + lag_1 + lag_2
    return lag


def calKpi_label(kpi_need_fdc, suffix=''):
    '''
    需要字段：sku_id, fdc_id, inv_his, inv_sim, fdc_sales_sim, sales_his_origin
    '''
    sku_cnt = len(np.unique(kpi_need_fdc.sku_id))
    sim_fdc_kpi=[]
    for label_fdcid, fdcdata in kpi_need_fdc.groupby(['fdc_id', 'label']):
        tmp_fdcid,label=label_fdcid[0],label_fdcid[1]
        fdc_kpi = defaultdict(lambda: defaultdict(float))
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid] = sum(fdcdata.inv_his > 0) / float(30 * sku_cnt)
        fdc_kpi['cr_sim'][tmp_fdcid] = sum(fdcdata.inv_sim > 0) / float(30 * sku_cnt)
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim)) <= 0 else float(
            np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin)) <= 0 else float(
            np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid] = -1 if float(fdc_kpi['ts_his'][tmp_fdcid]) <= 0 else float(
            fdc_kpi['ts_sim'][tmp_fdcid]) / float(fdc_kpi['ts_his'][tmp_fdcid])
        fdc_kpi['label'][tmp_fdcid]=label
        sim_fdc_kpi.append(pd.DataFrame(fdc_kpi))
    sim_fdc_kpi=pd.concat(sim_fdc_kpi)
    sim_fdc_kpi.columns = map(lambda x: x + suffix,list(sim_fdc_kpi.columns))
    sim_fdc_kpi.reset_index(inplace=True)
    sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    return sim_fdc_kpi


def calKpi(kpi_need_fdc, suffix=''):
    '''
    需要字段：sku_id, fdc_id, inv_his, inv_sim, sales_sim, sales_his_origin
    '''
    sku_cnt = len(np.unique(kpi_need_fdc.sku_id))
    fdc_kpi = defaultdict(lambda: defaultdict(float))
    for tmp_fdcid, fdcdata in kpi_need_fdc.groupby(['fdc_id']):
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid] = sum(fdcdata.inv_his > 0) / float(30 * sku_cnt)
        fdc_kpi['cr_sim'][tmp_fdcid] = sum(fdcdata.inv_sim > 0) / float(30 * sku_cnt)
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim)) <= 0 else float(
            np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin)) <= 0 else float(
            np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid] = -1 if float(fdc_kpi['ts_his'][tmp_fdcid]) <= 0 else float(
            fdc_kpi['ts_sim'][tmp_fdcid]) / float(fdc_kpi['ts_his'][tmp_fdcid])
    sim_fdc_kpi=pd.DataFrame(fdc_kpi)
    sim_fdc_kpi.columns = map(lambda x: x + suffix,list(sim_fdc_kpi.columns))
    sim_fdc_kpi.reset_index(inplace=True)
    sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    return sim_fdc_kpi


def count_diff():
    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    this_need = sim_all_sku_retail.loc[:,['dt', 'fdc_id', 'sku_id', 'allocation_retail_real', 'allocation_retail_cacl']]
    pd_count = pd.DataFrame()
    for key, value in this_need.groupby(['fdc_id', 'sku_id']):
        count_single = np.sum(value['allocation_retail_real'] != value['allocation_retail_cacl'])
        pd_dict = pd.DataFrame.from_dict({'fdc_id':key[0],'sku_id':key[1],'count':count_single}, orient='index').T
        pd_count = pd.concat([pd_count, pd_dict])
    pd_count = pd_count.loc[:,['fdc_id', 'sku_id', 'count']]
    pd_count_fdc = pd.DataFrame()
    for key, value in pd_count.groupby(['fdc_id']):
        count_fdc = np.sum(value['count'])
        pd_dict = pd.DataFrame.from_dict({'fdc_id':key,'count_fdc':count_fdc}, orient='index').T
        pd_count_fdc = pd.concat([pd_count_fdc, pd_dict])
    pd_count_sku = pd.DataFrame()
    for key, value in pd_count.groupby(['sku_id']):
        count_sku = np.sum(value['count'])
        pd_dict = pd.DataFrame.from_dict({'sku_id':key,'count_sku':count_sku}, orient='index').T
        pd_count_sku = pd.concat([pd_count_sku, pd_dict])
    pd_count = pd_count.merge(pd_count_sku, on=['sku_id'])
    pd_count = pd_count.merge(pd_count_fdc, on=['fdc_id'])
    pd_count.to_csv(read_path + os.sep + 'count_diff.csv', index=False)


def count_del():
    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    sales_his_origin_count = 0
    rdc_inv_count = 0
    count_both = 0
    count_either = 0
    sku_list = []
    for key, value in sim_all_sku_retail.groupby(['sku_id']):
        sales_his_origin = np.sum(value['sales_his_origin'])
        rdc_inv = np.sum(value['rdc_inv'])
        if sales_his_origin == 0:
            sales_his_origin_count += 1
            sku_list.append(key)
        if rdc_inv == 0:
            rdc_inv_count += 1
            sku_list.append(key)
        if (sales_his_origin == 0) and (rdc_inv == 0):
            count_both += 1
        if (sales_his_origin == 0) or (rdc_inv == 0):
            count_either += 1
    sku_list = list(set(sku_list))
    sku_list_pd = pd.DataFrame(sku_list, columns=['sku_id'])
    sku_retain = list(set(sim_all_sku_retail['sku_id'].unique()) - set(sku_list))
    sku_retain_pd = pd.DataFrame(sku_retain, columns=['sku_id'])
    sku_list_pd.to_csv(read_path + os.sep + 'sku_list_sim_del.csv', index=False)
    sku_retain_pd.to_csv(read_path + os.sep + 'sku_list_sim_retain.csv', index=False)
    sim_all_sku_retail_retain = sim_all_sku_retail.merge(sku_retain_pd, on=['sku_id'])
    sim_all_sku_retail_retain.to_csv(read_path + os.sep + 'sim_all_sku_retail_retain.csv', index=False)
    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    system_all_sku_retail_retain = system_all_sku_retail.merge(sku_list_pd, on=['sku_id'])
    system_all_sku_retail_retain.to_csv(read_path + os.sep + 'system_all_sku_retail_retain.csv', index=False)


def analysis_detail_01():
    def sale_label(x):
        if x == 0:
            return '0'
        elif (x >= 1) and (x <= 5):
            return '[1,5]'
        elif (x >= 6) and (x <= 10):
            return '[6,10]'
        elif (x >= 11) and (x <= 20):
            return '[11,20]'
        else:
            return '21'
    skulist = pd.read_csv(read_path + os.sep + 'skulist.csv')
    skulist['label'] = map(sale_label ,skulist['sales'].values)

    skulist_need = skulist[skulist['countalloctionreal'] > 0]
    # skulist_need = skulist_need.loc[:,['fdc_id','sku_id']]

    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    sim_all_sku_retail_need = sim_all_sku_retail.merge(skulist_need, on=['fdc_id','sku_id'])

    kpi_need = sim_all_sku_retail_need.copy()
    kpi_need_fdc = calKpi_label(kpi_need, suffix='_sim')

    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    sim_all_sku_retail_need = system_all_sku_retail.merge(skulist_need, on=['fdc_id', 'sku_id'])
    kpi_need_fdc_system = calKpi_label(sim_all_sku_retail_need, suffix='_system')

    kpi_total = kpi_need_fdc.merge(kpi_need_fdc_system, left_on=['fdc_id', 'label_sim'], right_on=['fdc_id', 'label_system'])

    # 计算分段的总量
    count_list = []
    for key, value in sim_all_sku_retail_need.groupby(['fdc_id']):
        tmp_count = dict(Counter(value['label']))
        tmp_count['fdc_id'] = key
        count_list.append(tmp_count)
    count_list_pd = pd.DataFrame.from_dict(count_list).T

    kpi_total.to_csv(read_path + os.sep + 'retainSku_fdc_kpi.csv', index=False)

    sim_all_sku_kpi = pd.read_table(read_path + os.sep + 'sim_all_sku_kpi.csv')
    sim_all_sku_kpi_need = sim_all_sku_kpi.merge(skulist_need, on=['fdc_id','sku_id'])
    sim_all_sku_kpi_need.to_csv(read_path + os.sep + 'sim_all_sku_kpi_retain.csv', index=False)


    # ------------------------------------- 简易的  -------------------------------------
    skulist = pd.read_csv(read_path + os.sep + 'skulist.csv')
    skulist_need = skulist[skulist['countalloctionreal'] > 0]
    skulist_need = skulist_need.loc[:,['fdc_id','sku_id']]

    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    sim_all_sku_retail_need = sim_all_sku_retail.merge(skulist_need, on=['fdc_id','sku_id'])

    kpi_need = sim_all_sku_retail_need.copy()
    kpi_need_fdc = calKpi(kpi_need, suffix='_sim')

    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    sim_all_sku_retail_need = system_all_sku_retail.merge(skulist_need, on=['fdc_id', 'sku_id'])
    kpi_need_fdc_system = calKpi(sim_all_sku_retail_need, suffix='_system')

    kpi_total = kpi_need_fdc.merge(kpi_need_fdc_system, on=['fdc_id'])
    kpi_total.to_csv(read_path + os.sep + 'retainSku_fdc_kpi_Total.csv', index=False)

    sim_all_sku_kpi = pd.read_table(read_path + os.sep + 'sim_all_sku_kpi.csv')
    sim_all_sku_kpi_need = sim_all_sku_kpi.merge(skulist_need, on=['fdc_id','sku_id'])
    sim_all_sku_kpi_need.to_csv(read_path + os.sep + 'sim_all_sku_kpi_retain.csv', index=False)

    # ====================================================================================
    # ====================================================================================
    # ====================================================================================

    skulist = pd.read_csv(read_path + os.sep + 'skulist.csv')
    skulist['label'] = map(sale_label ,skulist['sales'].values)

    skulist_need = skulist[skulist['countalloctionreal'] <= 0]

    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    sim_all_sku_retail_need = sim_all_sku_retail.merge(skulist_need, on=['fdc_id','sku_id'])

    kpi_need = sim_all_sku_retail_need.copy()
    kpi_need_fdc = calKpi_label(kpi_need, suffix='_sim')

    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    sim_all_sku_retail_need = system_all_sku_retail.merge(skulist_need, on=['fdc_id', 'sku_id'])
    kpi_need_fdc_system = calKpi_label(sim_all_sku_retail_need, suffix='_system')

    kpi_total = kpi_need_fdc.merge(kpi_need_fdc_system, left_on=['fdc_id', 'label_sim'], right_on=['fdc_id', 'label_system'])
    kpi_total.to_csv(read_path + os.sep + 'retainSku_fdc_kpi_del.csv', index=False)

    sim_all_sku_kpi = pd.read_table(read_path + os.sep + 'sim_all_sku_kpi.csv')
    sim_all_sku_kpi_need = sim_all_sku_kpi.merge(skulist_need, on=['fdc_id','sku_id'])
    sim_all_sku_kpi_need.to_csv(read_path + os.sep + 'sim_all_sku_kpi_retain_del.csv', index=False)


    # ------------------------------------- 简易的  -------------------------------------
    skulist = pd.read_csv(read_path + os.sep + 'skulist.csv')
    skulist_need = skulist[skulist['countalloctionreal'] <= 0]
    skulist_need = skulist_need.loc[:,['fdc_id','sku_id']]

    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    sim_all_sku_retail_need = sim_all_sku_retail.merge(skulist_need, on=['fdc_id','sku_id'])

    kpi_need = sim_all_sku_retail_need.copy()
    kpi_need_fdc = calKpi(kpi_need, suffix='_sim')

    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    sim_all_sku_retail_need = system_all_sku_retail.merge(skulist_need, on=['fdc_id', 'sku_id'])
    kpi_need_fdc_system = calKpi(sim_all_sku_retail_need, suffix='_system')

    kpi_total = kpi_need_fdc.merge(kpi_need_fdc_system, on=['fdc_id'])
    kpi_total.to_csv(read_path + os.sep + 'retainSku_fdc_kpi_Total_del.csv', index=False)

    sim_all_sku_kpi = pd.read_table(read_path + os.sep + 'sim_all_sku_kpi.csv')
    sim_all_sku_kpi_need = sim_all_sku_kpi.merge(skulist_need, on=['fdc_id','sku_id'])
    sim_all_sku_kpi_need.to_csv(read_path + os.sep + 'sim_all_sku_kpi_retain_del.csv', index=False)


def arrangeData():
    # 系统
    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    pd_data = pd.DataFrame()
    for key, value in system_all_sku_retail.groupby(['fdc_id', 'sku_id']):
        tmp_pd = value.copy()
        lag = lagFun(tmp_pd['sales_his_origin'].values)
        tmp_pd['sales_his_origin_sum'] = lag
        tmp_pd['sales_his_origin_flag'] = (tmp_pd['lop'] - tmp_pd['sales_his_origin_sum']) >= 0
        sr = np.sum(tmp_pd['sales_his_origin_flag'].values[:-2]) / 29
        T_value = -1 if np.sum(tmp_pd['sales_his_origin'].values[:-2]) == 0 else np.sum(tmp_pd['lop'].values[:-2]) / np.sum(tmp_pd['sales_his_origin'].values[:-2])
        tmp_pd2 = pd.DataFrame.from_dict({'fdc_id':key[0], 'sku_id':str(key[1]), 'SR':sr, 'T':T_value}, orient='index').T
        pd_data = pd.concat([pd_data, tmp_pd2])
    pd_data.index = range(len(pd_data))
    pd_data = pd_data.loc[:,['fdc_id', 'sku_id', 'SR', 'T']]
    pd_data = pd_data.sort_values(['sku_id', 'fdc_id'])
    pd_data.to_csv(read_path + os.sep + 'sr_T_values_system.csv', index=False)
    # 仿真
    system_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    pd_data = pd.DataFrame()
    for key, value in system_all_sku_retail.groupby(['fdc_id', 'sku_id']):
        tmp_pd = value.copy()
        lag = lagFun(tmp_pd['sales_his_origin'].values)
        tmp_pd['sales_his_origin_sum'] = lag
        tmp_pd['sales_his_origin_flag'] = (tmp_pd['lop'] - tmp_pd['sales_his_origin_sum']) >= 0
        sr = np.sum(tmp_pd['sales_his_origin_flag'].values[:-2]) / 29
        T_value = -1 if np.sum(tmp_pd['sales_his_origin'].values[:-2]) == 0 else np.sum(tmp_pd['lop'].values[:-2]) / np.sum(tmp_pd['sales_his_origin'].values[:-2])
        tmp_pd2 = pd.DataFrame.from_dict({'fdc_id':key[0], 'sku_id':str(key[1]), 'SR':sr, 'T':T_value}, orient='index').T
        pd_data = pd.concat([pd_data, tmp_pd2])
    pd_data.index = range(len(pd_data))
    pd_data = pd_data.loc[:,['fdc_id', 'sku_id', 'SR', 'T']]
    pd_data = pd_data.sort_values(['sku_id', 'fdc_id'])
    pd_data.to_csv(read_path + os.sep + 'sr_T_values_sim.csv', index=False)

