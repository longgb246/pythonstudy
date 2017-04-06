#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import time


# ======================================================================
# =                                 功能函数                           =
# ======================================================================
def lagFun(array):
    lag_0 = array
    lag_1 = np.array(list(lag_0[1:]) + [0])
    lag_2 = np.array(list(lag_1[1:]) + [0])
    lag = lag_0 + lag_1 + lag_2
    return lag


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
        print "[    Run Time   ] {3} is : {2} hours {0} min {1:.4f} s".format(min_d, sec_d, hor_d, name)
    else:
        print "[    Run Time   ] {2} is : {0} min {1:.4f} s".format(min_d, sec_d, name)


# ======================================================================
# =                                 计算函数                           =
# ======================================================================
def calKpi(kpi_need_fdc, suffix=''):
    '''
    需要字段：sku_id, fdc_id, inv_his, inv_sim, sales_sim, sales_his_origin
    '''
    fdc_kpi = defaultdict(lambda: defaultdict(float))
    for tmp_fdcid, fdcdata in kpi_need_fdc.groupby(['fdc_id']):
        sku_cnt = len(np.unique(fdcdata.sku_id))
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
        fdc_kpi['cr_his_new_0'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_0'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_his_new_12'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_12'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
    sim_fdc_kpi=pd.DataFrame(fdc_kpi)
    sim_fdc_kpi.columns = map(lambda x: x + suffix,list(sim_fdc_kpi.columns))
    sim_fdc_kpi.reset_index(inplace=True)
    sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    return sim_fdc_kpi


def calKpi_label(kpi_need_fdc, suffix=''):
    '''
    需要字段：sku_id, fdc_id, inv_his, inv_sim, fdc_sales_sim, sales_his_origin
    '''
    sim_fdc_kpi=[]
    for label_fdcid, fdcdata in kpi_need_fdc.groupby(['fdc_id', 'label']):
        sku_cnt = len(np.unique(fdcdata.sku_id))
        tmp_fdcid, label = label_fdcid[0], label_fdcid[1]
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
        fdc_kpi['cr_his_new_0'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_0'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_his_new_12'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_12'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
        sim_fdc_kpi.append(pd.DataFrame(fdc_kpi))
    sim_fdc_kpi=pd.concat(sim_fdc_kpi)
    sim_fdc_kpi.columns = map(lambda x: x + suffix,list(sim_fdc_kpi.columns))
    sim_fdc_kpi.reset_index(inplace=True)
    sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    sim_fdc_kpi.rename(columns={'label'+suffix: 'label'}, inplace=True)
    return sim_fdc_kpi


def getFileSuffix():
    '''
    读取 read_path 下所有的文件夹，并且抽取后缀名。
    '''
    all_files = []
    for each_file in os.listdir(read_path):
        if os.path.isdir(read_path + os.sep + each_file):
            tmp_file = each_file.split('_')
            if len(tmp_file) > 2:
                all_files.append(each_file)
    suffix = []
    for each_file in all_files:
        each_suffix = each_file.split('_')
        suffix_2 = ['std', 'L', 'lop', 'inverse']
        not_2 = True
        try:
            for each_suffix_2 in suffix_2:
                if each_suffix_2 in each_suffix[-2]:
                    not_2 = False
                    suffix.append('_' + reduce(lambda x, y: x + '_' + y, each_suffix[-2:]))
            if not_2:
                suffix.append('_' + each_suffix[-1])
        except:
            suffix.append(each_file)
    return all_files, suffix


def delSaleZero(sim_all_sku_retail, isSave=True):
    '''
    筛选 剔除总销量 为0的 SKU 和 FDC 组合
    '''
    sku_del_list = []
    # 2、---------------------------  ---------------------------
    for key, value in sim_all_sku_retail.groupby(['fdc_id', 'sku_id']):
        if np.sum(value['sales_his_origin']) == 0:
            sku_del_list.append(key)
    sku_all_list = map(lambda x: (x[0], x[1]), sim_all_sku_retail.loc[:, ['fdc_id', 'sku_id']].drop_duplicates().values)
    sku_keep_list = set(sku_all_list) - set(sku_del_list)
    sku_del_list_pd = pd.DataFrame(sku_del_list, columns=['fdc_id', 'sku_id'])
    sku_keep_list_pd = pd.DataFrame(list(sku_keep_list), columns=['fdc_id', 'sku_id'])
    if isSave:
        sku_del_list_pd.to_csv(save_path + os.sep + 'sku_del_list.csv', index=False)
        sku_keep_list_pd.to_csv(save_path + os.sep + 'sku_keep_list.csv', index=False)
    return sku_keep_list_pd, sku_del_list_pd


def combineKpi(kpi_list, suffix, kpi_band_list=[]):
    need_columns = ['cr_his_sim{0}'.format(suffix[0])] + ['cr_sim_sim{0}'.format(x) for x in suffix] + ['cr_sim_system{0}'.format(suffix[0])] + \
                   ['ito_his_sim{0}'.format(suffix[0])] + ['ito_sim_sim{0}'.format(x) for x in suffix] + ['ito_sim_system{0}'.format(suffix[0])] + \
                   ['ts_his_sim{0}'.format(suffix[0])] + ['ts_sim_sim{0}'.format(x) for x in suffix] + ['ts_sim_system{0}'.format(suffix[0])] + \
                   ['ts_rate_sim{0}'.format(x) for x in suffix] + ['ts_rate_system{0}'.format(suffix[0])]
                   # ['cr_his_new_0_sim{0}'.format(suffix[0]) for x in ['his'] ]
    kpi_list_1 = reduce(lambda x, y: x.merge(y, on=['fdc_id']), kpi_list)
    kpi_list_1_keep = kpi_list_1.loc[:,['fdc_id'] + need_columns]
    if kpi_band_list != []:
        kpi_list_2 = reduce(lambda x, y: x.merge(y, on=['fdc_id', 'label']), kpi_band_list)
        kpi_list_2_keep = kpi_list_2.loc[:,['fdc_id', 'label'] + need_columns]
    else:
        kpi_list_2_keep = []
    return kpi_list_1_keep, kpi_list_2_keep


def kpi_analysis(all_files=[], all_suffix=[]):
    # all_files = ['simulation_results_L_std', 'simulation_results_S', 'simulation_results_select', 'simulation_results_replace_std',
    #          'simulation_results_replace_std_7', 'simulation_results_replace_std_lop0.5', 'simulation_results_replace_std_lop1.5',
    #          'simulation_results_inverse']
    # all_suffix = ['_L_std', '_S', '_8', '_std', '_std_7', '_lop0.5', '_lop1.5', 'inverse']
    # 1、------------------------------ 获取文件名以及后缀 ------------------------------
    if all_files == []:
        files, suffix = getFileSuffix()
    else:
        files, suffix = all_files, all_suffix
    # 2、------------------------------ 读取band数据 ------------------------------------
    if ifBand:
        band = pd.read_table(band_path)
        band.columns = ['sku_id', 'label']
    kpi_list = []
    kpi_band_list = []
    for i, each_file in enumerate(files):
        print "[ Read the File ]: [ {0} ]".format(each_file)
        # 3.1、------------------------------ 读取文件 ------------------------------
        t1 = time.time()
        sim_all_sku_retail = pd.read_table(read_path + os.sep + each_file + os.sep + 'sim_all_sku_retail.csv')
        if i == 0:
            system_all_sku_retail = pd.read_table(read_path + os.sep + each_file + os.sep + 'system_all_sku_retail.csv')
        printRunTime(t1, 'Read File')
        # 3.2、------------------------------ 剔除销量为 0 ---------------------------
        if del_sale_zero:
            if i == 0:
                try:
                    sku_keep_list_pd = pd.read_csv(save_path + os.sep + 'sku_keep_list.csv')
                except:
                    print '[ Calculate Del ] Calculate Del sku for Sale is zero ...'
                    sku_keep_list_pd, sku_del_list_pd = delSaleZero(sim_all_sku_retail)
                    print '[ Calculate Del ] Finish the Calculate Del sku !'
            sim_all_sku_retail = sim_all_sku_retail.merge(sku_keep_list_pd, on=['fdc_id', 'sku_id'])
            system_all_sku_retail = system_all_sku_retail.merge(sku_keep_list_pd, on=['fdc_id', 'sku_id'])
        # 3.3、------------------------------ 计算 KPI -------------------------------
        print '[ Calculate KPI ] ...'
        t1 = time.time()
        sim_all_sku_retail_kpi = calKpi(sim_all_sku_retail, suffix='_sim' + suffix[i])
        system_all_sku_retail_kpi = calKpi(system_all_sku_retail, suffix='_system' + suffix[i])
        kpi_list.append(sim_all_sku_retail_kpi)
        if i == 0:
            kpi_list.append(system_all_sku_retail_kpi)
        if ifBand:
            sim_all_sku_retail_band = sim_all_sku_retail.merge(band, on=['sku_id'])
            system_all_sku_retail_band = system_all_sku_retail.merge(band, on=['sku_id'])
            sim_all_sku_retail_kpi_band = calKpi_label(sim_all_sku_retail_band, suffix='_sim' + suffix[i])
            system_all_sku_retail_kpi_band = calKpi_label(system_all_sku_retail_band, suffix='_system' + suffix[i])
            kpi_band_list.append(sim_all_sku_retail_kpi_band)
            if i == 0:
                kpi_band_list.append(system_all_sku_retail_kpi_band)
        printRunTime(t1, 'Calculate KPI')
    # 4、--------------------------- 合并 KPI ---------------------------
    kpi_list_keep, kpi_band_list_keep = combineKpi(kpi_list, suffix, kpi_band_list)
    # 5、--------------------------- 存储 KPI ---------------------------
    if add_true:
        kpi_list_keep.to_csv(save_path + os.sep + 'kpi_list_keep_keep_add.csv', index=False)
        if ifBand:
            kpi_band_list_keep.to_csv(save_path + os.sep + 'kpi_band_list_keep_add.csv', index=False)
    else:
        kpi_list_keep.to_csv(save_path + os.sep + 'kpi_list_keep.csv', index=False)
        if ifBand:
            kpi_band_list_keep.to_csv(save_path + os.sep + 'kpi_band_list_keep.csv', index=False)


# ======================================================================
# =                                 配置参数                            =
# ======================================================================
read_path = r'D:\Lgb\WorkFiles\FDC_UNION_ALLOCATION\news\fdcall_std_7'
band_path = r'D:\Lgb\WorkFiles\FDC_UNION_ALLOCATION\analysis_3_policy\SKUABCband_20170330153011.csv'
save_path = r'D:\Lgb\WorkFiles\FDC_UNION_ALLOCATION\news\fdcall_std_7\Result'
del_sale_zero = True        # 是否剔除销量为 0
ifBand = True               # 是否计算分 band 的值
add_true = False            # 是否单独储存某文件夹下的 kpi，如果 True 则需 all_files、all_suffix 值， False 则计算 read_path 下所有文件夹的 kpi
all_files=[]
all_suffix=[]


if __name__ == '__main__':
    kpi_analysis()

