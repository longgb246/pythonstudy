# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import cPickle as pickle
from collections import defaultdict
import time


def getdaterange(start_date,end_date):
    '''
    生成日期，格式'yyyy-mm-dd'
    '''
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date).values)
    return date_range


def printruntime(t1, name):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d >0:
        print '[   Run Time   ] ({3}) is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print '[   Run Time   ] ({2}) is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


start_date = '2016-10-02'
end_date = '2016-10-31'

path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total_actual'
read_path = path + os.sep + 'org_data'
sku_data_path = read_path + os.sep + 'total_sku'
fdc_data_path = read_path + os.sep + 'fdc_data.pkl'
order_data_path = read_path + os.sep + 'order_data.pkl'
sale_data_path = read_path + os.sep + 'total_sale'
# 存储路径
save_path = path
sku_save_path = path + os.sep + 'total_sku'

if os.path.exists(sku_save_path) == False:
    os.mkdir(sku_save_path)


# 'white_flag','date_s', 'sku_id', 'dc_id', 'inv'
# 1、Sku 需要
date_range = getdaterange(start_date, end_date)


def transfer_sku():
    '''
    生成 sku 的每日数据，safe_max.pkl、all_sku_list.pkl
    '''
    all_sku_list = []
    safe_max = []
    for each in date_range:
        tmp_data_pkl = open(sku_data_path + os.sep + '{0}.pkl'.format(each))
        tmp_data_pd = pickle.load(tmp_data_pkl)
        tmp_data_pkl.close()
        tmp_data_pd.columns = [ 'sku_id', 'mean_sales', 'variance', 'ofdsales', 'inv', 'white_flag', 'white_flag_01',
                                'date_s', 'dc_id', 'variance_ofdsales', 'std']
        tmp_data_pd_cal = tmp_data_pd.loc[:,['sku_id', 'date_s', 'dc_id', 'mean_sales']]
        safe_max.append(tmp_data_pd_cal)
        tmp_data_pd_save = tmp_data_pd.loc[:,['white_flag','date_s', 'sku_id', 'dc_id', 'inv']]
        all_sku_list.extend(list(tmp_data_pd_save['sku_id']))
        pickle.dump(tmp_data_pd_save, open(sku_save_path + os.sep + '{0}.pkl'.format(each), 'w'))
    safe_max_pd = pd.concat(safe_max)
    safe_max_pd['safe_qtty'] = map(lambda x: np.mean(eval(x)) * 4, safe_max_pd['mean_sales'].values)
    safe_max_pd['max_qtty'] = safe_max_pd['safe_qtty'] * 2
    safe_max_pd = safe_max_pd.loc[:, ['sku_id', 'date_s', 'dc_id', 'safe_qtty', 'max_qtty']]
    pickle.dump(safe_max_pd, open(save_path + os.sep + 'safe_max.pkl', 'w'))
    all_sku_list = list(set(all_sku_list))
    pickle.dump(all_sku_list, open(save_path + os.sep + 'all_sku_list.pkl', 'w'))


def transfer_fdc():
    '''
    生成 fdc_alt.pkl、fdc_alt_prob.pkl
    '''
    pkl_fdc = open(fdc_data_path)
    allocation_fdc_data = pickle.load(pkl_fdc)
    pkl_fdc.close()
    allocation_fdc_data.columns = ['org_from', 'org_to', 'actiontime_max', 'alt', 'alt_cnt']
    fdc_01 = allocation_fdc_data.groupby(['org_from', 'org_to', 'alt']).sum()
    fdc_01 = fdc_01.reset_index()
    fdc_02 = allocation_fdc_data['alt_cnt'].groupby([allocation_fdc_data['org_from'], allocation_fdc_data['org_to']]).sum()
    fdc_02 = fdc_02.reset_index()
    fdc_alt = pd.merge(fdc_01, fdc_02, on=['org_from', 'org_to'])
    fdc_alt.columns = ['org_from', 'org_to', 'alt', 'alt_cnt', 'alt_all_cnt']
    fdc_alt['alt_prob'] = fdc_alt['alt_cnt'] / fdc_alt['alt_all_cnt']
    allocation_fdc_data = fdc_alt
    allocation_fdc_data.columns = ['org_from', 'dc_id', 'alt', 'alt_cnt', 'alt_all_cnt', 'alt_prob']
    allocation_fdc_data['dc_id'] = map(lambda x: str(int(x)), allocation_fdc_data['dc_id'].values)
    allocation_fdc_data = allocation_fdc_data[allocation_fdc_data['org_from'] == 316]
    fdc_alt = defaultdict(list)
    fdc_alt_prob = defaultdict(list)
    for index, row in allocation_fdc_data.iterrows():
        if row['dc_id'] in fdc_alt:
            try:
                tmp = eval(row['alt'])
                fdc_alt[row['dc_id']].append(tmp)
            except:
                pass
        else:
            try:
                tmp = eval(row['alt'])
                fdc_alt[row['dc_id']] = [tmp]
            except:
                pass
        if row['dc_id'] in fdc_alt_prob:
            try:
                tmp = row['alt_prob']
                fdc_alt_prob[row['dc_id']].append(tmp)
            except:
                pass
        else:
            try:
                tmp = row['alt_prob']
                fdc_alt_prob[row['dc_id']] = [tmp]
            except:
                pass
    pickle.dump(fdc_alt, open(save_path + os.sep + 'fdc_alt.pkl', 'w'))
    pickle.dump(fdc_alt_prob, open(save_path + os.sep + 'fdc_alt_prob.pkl', 'w'))


def transfer_order():
    '''
    生成 order_data.pkl
    '''
    pkl_order = open(order_data_path)
    allocation_order_data = pickle.load(pkl_order)
    pkl_order.close()
    allocation_order_data.columns = ['arrive_time', 'item_sku_id', 'arrive_quantity', 'dc_id']
    allocation_order_data = allocation_order_data.loc[:,['arrive_time', 'item_sku_id', 'arrive_quantity']]
    pickle.dump(allocation_order_data, open(save_path + os.sep + 'order_data.pkl', 'w'))


def transfer_sale():
    '''
    生成 sale_data.pkl
    '''
    pkl_sale = []
    for p in date_range:
        pkl_sale_mid = open(sale_data_path + os.sep + p + '.pkl')
        mid_allocation_sale_data = pickle.load(pkl_sale_mid)
        pkl_sale.append(mid_allocation_sale_data)
        pkl_sale_mid.close()
    allocation_sale_data = pd.concat(pkl_sale)
    allocation_sale_data.columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id', 'item_sku_id',
                                    'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'white_flag_01',
                                    'item_third_cate_cd',
                                    'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    allocation_sale_data = allocation_sale_data.loc[:,['dc_id', 'date_s', 'item_sku_id', 'parent_sale_ord_id', 'sale_ord_tm', 'sale_qtty']]
    pickle.dump(allocation_sale_data, open(save_path + os.sep + 'sale_data.pkl', 'w'))


if __name__ == '__main__':
    print 'Start Run !'
    t2 = time.time()
    t1 = time.time()
    transfer_order()
    printruntime(t1, 'Order 完成')
    t1 = time.time()
    transfer_fdc()
    printruntime(t1, 'FDC 完成')
    t1 = time.time()
    transfer_sale()
    printruntime(t1, 'Sale 完成')
    t1 = time.time()
    transfer_sku()
    printruntime(t1, 'Sku 完成')
    printruntime(t2, ' Run All programe ! ')


