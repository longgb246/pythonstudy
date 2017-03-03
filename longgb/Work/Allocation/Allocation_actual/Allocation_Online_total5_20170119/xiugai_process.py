#-*- coding:utf-8 -*-
import os
import pandas as pd
import cPickle as pickle


path = r'D:\Lgb\data_sz\allocation_sim_online_total5\fdv_inv'
save_path = r'D:\Lgb\data_local\allocation_sim_online_total5'


def find_sku():
    # 发现SKU的open_po有问题
    fdc_inv_pd_02 = pd.read_csv(r'D:\Lgb\data_sz\allocation_sim_online_total5\fdc_inv_intial_02\fdc_inv_pd.csv')
    # fdc_inv_pd_02.columns
    tmp_fdc_inv_pd_02 = fdc_inv_pd_02[(fdc_inv_pd_02['date_s'] == '2016-10-02')&(fdc_inv_pd_02['fdc'] == 628)]
    print tmp_fdc_inv_pd_02[tmp_fdc_inv_pd_02['open_po'] != 0]
    # 1964035, 851977
    sku_list = list(set(tmp_fdc_inv_pd_02[tmp_fdc_inv_pd_02['open_po'] != 0]['sku']))
    print len(sku_list)
    print sku_list


def find_csv_analysis(sku):
    # 取样本SKU分析
    fdc_inv_pd = pd.read_csv(path + os.sep + 'fdc_inv_pd.csv')
    fdc_inv_pd = fdc_inv_pd[fdc_inv_pd['sku'] == sku]
    fdc_inv_pd = fdc_inv_pd[fdc_inv_pd['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd.to_csv(save_path + os.sep + 'fdc_inv_pd.csv', index=False)
    #
    fdc_inv_pd_02 = pd.read_csv(r'D:\Lgb\data_sz\allocation_sim_online_total5\fdc_inv_intial_02\fdc_inv_pd.csv')
    fdc_inv_pd_02 = fdc_inv_pd_02[fdc_inv_pd_02['sku'] == sku]
    fdc_inv_pd_02 = fdc_inv_pd_02[fdc_inv_pd_02['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd_02.to_csv(save_path + os.sep + 'fdc_inv_pd_02.csv', index=False)
    #
    fdc_inv_pd_03 = pd.read_csv(r'D:\Lgb\data_sz\fdc_inv\fdc_inv_pd.csv')
    fdc_inv_pd_03 = fdc_inv_pd_03[fdc_inv_pd_03['sku'] == sku]
    fdc_inv_pd_03 = fdc_inv_pd_03[fdc_inv_pd_03['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd_03.to_csv(save_path + os.sep + 'fdc_inv_pd_03.csv', index=False)
    #
    fdc_inv_pd_01_two = pd.read_csv(r'D:\Lgb\data_sz\fdc_inv\two_day_debug\01\01_fdc\fdc_inv_pd.csv')
    fdc_inv_pd_01_two = fdc_inv_pd_01_two[fdc_inv_pd_01_two['sku'] == sku]
    fdc_inv_pd_01_two = fdc_inv_pd_01_two[fdc_inv_pd_01_two['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd_01_two.to_csv(save_path + os.sep + 'fdc_inv_pd_01_two.csv', index=False)
    #
    fdc_inv_pd_02_two = pd.read_csv(r'D:\Lgb\data_sz\fdc_inv\two_day_debug\02\02_fdc\fdc_inv_pd.csv')
    fdc_inv_pd_02_two = fdc_inv_pd_02_two[fdc_inv_pd_02_two['sku'] == sku]
    fdc_inv_pd_02_two = fdc_inv_pd_02_two[fdc_inv_pd_02_two['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd_02_two.to_csv(save_path + os.sep + 'fdc_inv_pd_02_two.csv', index=False)
    #
    fdc_inv_pd_01_two_update = pd.read_csv(r'D:\Lgb\data_sz\fdc_inv\two_day_debug\01_fdcinv\fdc_inv_pd.csv')
    fdc_inv_pd_01_two_update = fdc_inv_pd_01_two_update[fdc_inv_pd_01_two_update['sku'] == sku]
    fdc_inv_pd_01_two_update = fdc_inv_pd_01_two_update[fdc_inv_pd_01_two_update['fdc'] == 628].sort_values('date_s')

    tmp_aa = fdc_inv_pd_01_two_update[(fdc_inv_pd_01_two_update['date_s'] == '2016-10-03')&(fdc_inv_pd_01_two_update['arrive_quantity'] > 0)]
    list(tmp_aa['sku'])
    fdc_inv_pd_01_two_update[fdc_inv_pd_01_two_update['sku'] == 108469].sort_values(['fdc','date_s'])

    fdc_inv_pd_01_two_update.to_csv(save_path + os.sep + 'fdc_inv_pd_01_two_update.csv', index=False)
    #
    fdc_inv_pd_02_two_update = pd.read_csv(r'D:\Lgb\data_sz\fdc_inv\two_day_debug\02_fdcinv\fdc_inv_pd.csv')
    fdc_inv_pd_02_two_update = fdc_inv_pd_02_two_update[fdc_inv_pd_02_two_update['sku'] == sku]
    fdc_inv_pd_02_two_update = fdc_inv_pd_02_two_update[fdc_inv_pd_02_two_update['fdc'] == 628].sort_values('date_s')
    fdc_inv_pd_02_two_update.to_csv(save_path + os.sep + 'fdc_inv_pd_02_two_update.csv', index=False)


# pkl 文件
fdc_inv_intial_01 = pickle.load(open(path + os.sep + 'fdc_inv_intial' + os.sep + '2016-10-01.pkl'))
fdc_inv_intial_02 = pickle.load(open(path + os.sep + 'fdc_inv_intial' + os.sep + '2016-10-02.pkl'))

find_csv_analysis(1964035)
print 'over!'


# 1964035




