# coding=utf-8
from scipy.stats import rv_discrete, norm
import numpy as np
import math
import pandas as pd
from collections import defaultdict, OrderedDict
import time, datetime
import copy
import pickle
import logging
import os


fdc_forecast_sales = 1
fdc_forecast_std = 1
fdc_inv = 1
white_list_dict = 1
fdc = 1
order_list = 1
date_range = 1
orders_retail = 1
all_sku_list = 1
logger = 1
save_data_path = 1
union_white_list = 1

rdc_inv = defaultdict(int)
rdc_inv.update(rdc_inv)

fdc_alt_prob = defaultdict(list)
fdc_alt = defaultdict(list)
fdc_alt.update(fdc_alt)
fdc_alt_prob.update(fdc_alt_prob)

simu_orders_retail = copy.deepcopy(orders_retail)
fdc_simu_orders_retail = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
orders_retail_type = defaultdict(str)
sku_fdc_sales = defaultdict(int)
sku_rdc_sales = defaultdict(int)


fdc_allocation = defaultdict(float)
allocation_retail= defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def gene_whitelist( d):
    '''获取该时间点的白名单,调用户一次刷新一次，只保存最新的白名单列表'''
    white_list = defaultdict(list)
    union_white_list = []
    for f in fdc:
        for k, v in white_list_dict[f].items():
            if k == d:
                white_list[f].extend(v)  # list[]
                union_white_list.extend(v)
    union_white_list = list(set(union_white_list))

def cacl_rdc_inv( d):
    '''  补货逻辑 #更新RDC库存,RDC库存的更新按照实际订单情况进行更新，rdc{index:库存量}'''
    for s in all_sku_list:
        #sku的长度>3 否则为非法的sku
        if len(str(s)) < 3:
            continue
        index = gene_index('rdc', s, d)
        rdc_inv[index] = rdc_inv[index] + order_list[d].get(s, 0)

def calc_lop( sku, fdc, date_s, cr=0.99):
    '''    #计算某个FDC的某个SKU的补货点'''
    # sku的FDC销量预测，与RDC的cv系数
    if sku not in union_white_list:
        return 0
    index = gene_index(fdc, sku, date_s)
    sku_sales = eval(fdc_forecast_sales[index])
    sku_sales_mean = np.mean(sku_sales)
    safe_qtty = sku_sales_mean * 4
    lop = safe_qtty
    return lop

def calc_replacement( sku, fdc, date_s, sku_lop, bp=10, cr=0.99):
    '''
    #计算某个FDC的SKU的补货量
    计算补货量补货量为lop+bp-在途-当前库存
    '''
    # sku的FDC销量预测，与RDC的cv系数
    index = gene_index(fdc, sku, date_s)
    sku_sales = eval(fdc_forecast_sales[index])
    sku_sales_mean = np.mean(sku_sales)
    max_qtty = sku_sales_mean * 8
    inv = fdc_inv[index]['inv']
    open_on = fdc_inv[index]['open_po']
    lop_replacement = max(max_qtty - inv - open_on,0)
    if lop_replacement <= 10:
        pass
    else:
        div_num, mod_num = divmod(lop_replacement, 10)
        if mod_num <= 2:
            lop_replacement = div_num * 10
        elif mod_num <= 7:
            lop_replacement = div_num * 10 + 5
        else:
            lop_replacement = (div_num + 1) * 10
    return np.floor(lop_replacement)

def calc_sku_allocation( d):
    '''
    首先将当日到达量加到当日库存中
    @sku
    @调拨量
    ----输出---
    @date:日期
    @sku:sku
    @fdc:fdc
    @allocation：调拨量
    '''
    t1 = time.time()
    i_num = 0
    for s in union_white_list:
        i_num += 1
        if i_num % 1000 == 0:
            print i_num
        fdc_replacement = defaultdict(int)
        for f in fdc:
            if s not in white_list[f]:
                fdc_replacement[f] = 0
            else:
                lop_tmp = calc_lop(s, f, d)
                index = gene_index(f, s, d)
                if (fdc_inv[index]['inv'] + fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po']) < lop_tmp:
                    fdc_replacement[f] = calc_replacement(s, f, d, lop_tmp)
                else:
                    fdc_replacement[f] = 0
        index_rdc = gene_index('rdc', s, d)
        rdc_inv_avail = max(np.min([rdc_inv[index_rdc] - 12, np.floor(rdc_inv[index_rdc] * 0.2)]), 0)
        allocation_retail[d][s]['rdc'] = rdc_inv[index_rdc]
        for f in ['630', '658', '628']:
            index_fdc = gene_index(f, s, d)
            allocation_retail[d][s]['calc_' + f] = fdc_replacement[f]
            fdc_inv_avail = np.min([fdc_replacement[f], rdc_inv_avail])
            allocation_retail[d][s]['real_' + f] = fdc_inv_avail
            fdc_allocation[index_fdc] = fdc_inv_avail
            rdc_inv[index_rdc] = rdc_inv[index_rdc] - fdc_inv_avail
            rdc_inv_avail -= fdc_inv_avail
    printruntime(t1, 'calc_sku_allocation')             # [   Run Time   ] (calc_sku_allocation) is : 6.0 min 11.6580 s

# fdc_inv['2016-10-026281964035']
# fdc_inv['2016-10-026281311040']
# fdc_inv['2016-10-026281835048']
# fdc_inv['2016-10-036281835048']
# fdc_inv['2016-10-046281835048']


def gene_index( fdc, sku, date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s) + str(fdc) + str(sku)

def gene_alt( f):
    '''
    生成对应的调拨时长，用以更新FDC的库存
    '''
    fdc_vlt = fdc_alt[f]
    fdc_vlt_porb = fdc_alt_prob[f]
    # 如果没有对应的调拨时长，默认为3天
    if len(fdc_vlt) == 0:
        return 3
    alt_distribution = rv_discrete(values=(fdc_vlt, fdc_vlt_porb))
    return alt_distribution.rvs()

def calc_fdc_allocation( d, f):
    '''
    ------输入数据格式----
    @date:日期 20161129,字符串格式
    @fdc:fdc
    ------输出数据格式,dataframe-----
    @date:日期
    @sku:sku
    @fdc:fdc
    @inv:当前库存
    @allocation：调拨量
    @open_po:在途量
    @arrive_quantity:当日到达量
    '''
    # 计算补货点，判断补货量
    # 假设一个RDC-FDC同一天的调拨 运达实际相同
    t1 = time.time()
    for s in union_white_list:
        # s = union_white_list[0]              s = 1311040  f = '628'
        index = gene_index(f, s, d)
        if fdc_allocation[index] > 0:
            # print index
            # break
            fdc_inv[index]['allocation'] = fdc_allocation[index]
            alt=-1
            while (alt<1 or alt >10):
                alt = gene_alt(f)
            c = 0
            format_date = '%Y-%m-%d'
            while c < alt:
                date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(c)
                date_s_c = date_tmp.strftime('%Y-%m-%d')
                index_tmp = gene_index(f, s, date_s_c)
                fdc_inv[index_tmp]['open_po'] = fdc_inv[index]['allocation'] + fdc_inv[index_tmp]['open_po']
                c += 1
            date_alt = datetime.datetime.strptime(d, format_date) + datetime.timedelta(alt)
            date_s_alt = date_alt.strftime(format_date)
            index_1 = gene_index(f, s, date_s_alt)
            fdc_inv[index_1]['arrive_quantity'] = fdc_inv[index]['allocation'] + fdc_inv[index_1]['arrive_quantity']
        fdc_inv[index]['inv'] = fdc_inv[index]['inv'] + \
                                     max(fdc_inv[index]['arrive_quantity']-fdc_inv[index]['cons_open_po'],0)
        fdc_inv[index]['cons_open_po']=max(fdc_inv[index]['cons_open_po']-fdc_inv[index]['arrive_quantity'],0)
    printruntime(t1, 'fdc_inv')     # [   Run Time   ] (fdc_inv) is : 0.0 min 2.0980 s


def mkdir_save(self):
    save_data_path_org = save_data_path + 'org_data'
    if os.path.exists(save_data_path_org) == False:
        os.mkdir(save_data_path_org)
    save_data_path_fdc_inv = save_data_path + 'fdc_inv'
    if os.path.exists(save_data_path_fdc_inv) == False:
        os.mkdir(save_data_path_fdc_inv)
    save_data_path_fdc_allocation = save_data_path + 'fdc_allocation'
    if os.path.exists(save_data_path_fdc_allocation) == False:
        os.mkdir(save_data_path_fdc_allocation)
    save_data_path_rdc_inv = save_data_path + 'rdc_inv'
    if os.path.exists(save_data_path_rdc_inv) == False:
        os.mkdir(save_data_path_rdc_inv)
    save_data_path_white_list = save_data_path + 'white_list'
    if os.path.exists(save_data_path_white_list) == False:
        os.mkdir(save_data_path_white_list)
    save_data_path_order_list = save_data_path + 'order_list'
    if os.path.exists(save_data_path_order_list) == False:
        os.mkdir(save_data_path_order_list)
    save_data_path_white_list = save_data_path + 'white_list'
    if os.path.exists(save_data_path_white_list) == False:
        os.mkdir(save_data_path_white_list)
    save_data_path_orders_retail = save_data_path + 'orders_retail'
    if os.path.exists(save_data_path_orders_retail) == False:
        os.mkdir(save_data_path_orders_retail)
    save_data_path_simu_orders_retail = save_data_path + 'simu_orders_retail'
    if os.path.exists(save_data_path_simu_orders_retail) == False:
        os.mkdir(save_data_path_simu_orders_retail)
    save_data_path_fdc_simu_orders_retail = save_data_path + 'fdc_simu_orders_retail'
    if os.path.exists(save_data_path_fdc_simu_orders_retail) == False:
        os.mkdir(save_data_path_fdc_simu_orders_retail)

def save_oneday(date_s):
    logger.info('Save the median Data : {0} ...'.format(date_s))
    pickle.dump(dict(fdc_inv),
                open(save_data_path_fdc_inv + os.sep + 'fdc_inv_{0}.pkl'.format(date_s), 'w'))
    pickle.dump(dict(fdc_allocation),
                open(save_data_path_fdc_allocation + os.sep + 'fdc_allocation_{0}.pkl'.format(date_s), 'w'))
    pickle.dump(dict(rdc_inv),
                open(save_data_path_rdc_inv + os.sep + 'rdc_inv_{0}.pkl'.format(date_s), 'w'))
    with open(save_data_path_order_list + os.sep + 'order_list_{0}.txt'.format(date_s), 'w') as ol:
        for k, v in order_list.items():
            for k1, v1 in v.items():
                ol.write(str(k))
                ol.write('\t')
                ol.write(str(k1))
                ol.write('\t')
                ol.write(str(v1))
            ol.write('\n')
    with open(save_data_path_orders_retail + os.sep + 'orders_retail_{0}.txt'.format(date_s), 'w') as orl:
        for k, v in orders_retail.items():
            for k1, v1 in v.items():
                for k2, v2 in v1.items():
                    orl.write(str(k))
                    orl.write('\t')
                    orl.write(str(k1))
                    orl.write('\t')
                    orl.write(str(k2))
                    orl.write('\t')
                    orl.write(str(v2))
            orl.write('\n')
    try:
        with open(save_data_path_simu_orders_retail + os.sep + 'simu_orders_retail_{0}.txt'.format(date_s),
                  'w') as orl:
            for k, v in simu_orders_retail.items():
                for k1, v1 in v.items():
                    for k2, v2 in v1.items():
                        orl.write(str(k))
                        orl.write('\t')
                        orl.write(str(k1))
                        orl.write('\t')
                        orl.write(str(k2))
                        orl.write('\t')
                        orl.write(str(v2))
                orl.write('\n')
    except:
        print 'simu order  in the except'
    try:
        with open(save_data_path_fdc_simu_orders_retail + os.sep + 'fdc_simu_orders_retail_{0}.txt'.format(
                date_s), 'w') as orl:
            for k, v in fdc_simu_orders_retail.items():
                for k1, v1 in v.items():
                    for k2, v2 in v1.items():
                        orl.write(str(k))
                        orl.write('\t')
                        orl.write(str(k1))
                        orl.write('\t')
                        orl.write(str(k2))
                        orl.write('\t')
                        orl.write(str(v2))
                orl.write('\n')
    except:
        print 'in the except'
    logger.info('Save the median Data : {0} . Finish !'.format(date_s))

def OrdersSimulation(self):
    rdc_fdc=['630', '658', '628','316']
    for d in date_range:
        d = date_range[0]
        gene_whitelist(d)
        cacl_rdc_inv(d)
        calc_sku_allocation(d)
        all_rdc_orders_reatail = {}
        top_n_min_orders_retail = {}
        for f in rdc_fdc:
            # f = rdc_fdc[0]
            # f = rdc_fdc[1]
            # f = rdc_fdc[2]
            # f = rdc_fdc[3]
            calc_fdc_allocation(d, f)
            tmp_order_retail = orders_retail[f + d]
            all_rdc_orders_reatail[f] = copy.deepcopy(sorted(tmp_order_retail.items(), key=lambda d: d[0]))
            top_n_min_orders_retail[f] = all_rdc_orders_reatail[f].pop(0)
        while top_n_min_orders_retail:
            o = copy.deepcopy(sorted(top_n_min_orders_retail.items(), key=lambda t: t[1][0]).pop(0))
            f = o[0]
            del top_n_min_orders_retail[f]
            order_index = o[1][0]
            sku_state = []
            for s in dict(o[1][1]).items():
                index = gene_index(f, s[0], d)
                index_rdc = gene_index('rdc', s[0], d)
                tmp = defaultdict(int)
                fdc_useage_inv=fdc_inv[index]['inv']+fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po']
                if s[0] not in white_list[f]:
                    if s[1] >= rdc_inv[index_rdc] >= 0:
                        tmp['rdc'] = 1
                        tmp['fdc_rdc'] = 1
                    else:
                        tmp['rdc'] = 1
                        tmp['fdc_rdc'] = 1
                elif s[1] > fdc_useage_inv:
                    if s[1] > rdc_inv[index_rdc]:
                        if s[1] > rdc_inv[index_rdc] + fdc_useage_inv:
                            tmp['fdc_rdc'] = 1
                        else:
                            tmp['fdc_rdc'] = 1
                    else:
                        tmp['rdc'] = 1
                        tmp['fdc_rdc'] = 1
                else:
                    tmp['fdc'] = 1
                    tmp['fdc_rdc'] = 1
                    if s[1] <= rdc_inv[index_rdc]:
                        tmp['rdc'] = 1
                sku_state.append(copy.deepcopy(tmp))
            flag_fdc = min([c['fdc'] for c in sku_state])
            flag_rdc = min([c['rdc'] for c in sku_state])
            flag_fdc_rdc = min([c['fdc_rdc'] for c in sku_state])
            if flag_fdc == 1:
                orders_retail_type[o[1][0]] = 'fdc'
                for s in dict(o[1][1]).items():
                    index = gene_index(f, s[0], d)
                    fdc_simu_orders_retail[f + d][order_index][s[0]] = orders_retail[f + d][order_index][s[0]]
                    if s[1]>fdc_inv[index]['inv']:
                        fdc_inv[index]['cons_open_po']=fdc_inv[index]['cons_open_po']+np.min([s[1]-fdc_inv[index]['inv'],
                                                                                                    fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po']])
                    fdc_inv[index]['inv'] = fdc_inv[index]['inv'] - min(s[1], fdc_inv[index]['inv'])
            elif flag_rdc == 1:
                orders_retail_type[o[1][0]] = 'rdc'
                for s in dict(o[1][1]).items():
                    index = gene_index('rdc', s[0], d)
                    simu_orders_retail[f + d][order_index][s[0]] = min(rdc_inv[index_rdc], s[1])
                    rdc_inv[index] = rdc_inv[index] - min(s[1], rdc_inv[index])
            elif flag_fdc_rdc == 1:
                orders_retail_type[o[1][0]] = 'fdc_rdc'
                for s in dict(o[1][1]).items():
                    index = gene_index(f, s[0], d)
                    rdc_index = gene_index('rdc', s[0], d)
                    fdc_simu_orders_retail[f + d][order_index][s[0]] = min(orders_retail[f + d][order_index][s[0]],
                                                                                fdc_inv[index]['inv']+fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    simu_orders_retail[f + d][order_index][s[0]] = min(orders_retail[f + d][order_index][s[0]],fdc_inv[index]['inv']
                                                                        +fdc_inv[index]['open_po']
                                                                        -fdc_inv[index]['cons_open_po']
                                                                        + rdc_inv[rdc_index])
                    if s[1]>fdc_inv[index]['inv']:
                        fdc_inv[index]['cons_open_po']=fdc_inv[index]['cons_open_po']+min(s[1]-fdc_inv[index]['inv'],
                                                                                                    fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    sku_gap = s[1] - fdc_inv[index]['inv']
                    fdc_inv[index]['inv'] = 0 if sku_gap >= 0 else abs(sku_gap)
                    sku_gap = s[1] - (fdc_inv[index]['inv']+fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    rdc_inv[rdc_index] = rdc_inv[rdc_index] if sku_gap < 0 else max(
                        rdc_inv[rdc_index] - sku_gap, 0)
            else:
                orders_retail_type[o[1][0]] = 'other'
                for s in dict(o[1][1]).items():
                    index = gene_index(f, s[0], d)
                    rdc_index = gene_index('rdc', s[0], d)
                    fdc_simu_orders_retail[f + d][order_index][s[0]] = min(orders_retail[f + d][order_index][s[0]],
                                                                                fdc_inv[index]['inv']+fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    simu_orders_retail[f + d][order_index][s[0]] = min(orders_retail[f + d][order_index][s[0]],fdc_inv[index]['inv']
                                                                            +fdc_inv[index]['open_po']
                                                                            -fdc_inv[index]['cons_open_po']
                                                                            + rdc_inv[rdc_index])
                    if s[1]>fdc_inv[index]['inv']:
                        fdc_inv[index]['cons_open_po']=fdc_inv[index]['cons_open_po']+min(s[1]-fdc_inv[index]['inv'],
                                                                                                    fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    sku_gap = s[1] - fdc_inv[index]['inv']
                    fdc_inv[index]['inv'] = 0 if sku_gap >= 0 else abs(sku_gap)
                    sku_gap = s[1] - (fdc_inv[index]['inv']+fdc_inv[index]['open_po']-fdc_inv[index]['cons_open_po'])
                    rdc_inv[rdc_index] = rdc_inv[rdc_index] if sku_gap < 0 else max(
                        rdc_inv[rdc_index] - sku_gap, 0)
            if all_rdc_orders_reatail.has_key(f):
                if len(all_rdc_orders_reatail[f]) > 0:
                    top_n_min_orders_retail[f] = all_rdc_orders_reatail[f].pop(0)
                else:
                    del all_rdc_orders_reatail[f]
        for f in fdc:
            for s in white_list[f]:
                format_date = '%Y-%m-%d'
                date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
                date_s_c = date_tmp.strftime('%Y-%m-%d')
                index_next = gene_index(f, s, date_s_c)
                index = gene_index(f, s, d)
                fdc_inv[index_next]['inv'] = fdc_inv[index]['inv']
                fdc_inv[index_next]['cons_open_po'] = fdc_inv[index]['cons_open_po']
        for s in all_sku_list:
            format_date = '%Y-%m-%d'
            date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
            date_s_c = date_tmp.strftime('%Y-%m-%d')
            index_next = gene_index('rdc', s, date_s_c)
            index = gene_index('rdc', s, d)
            rdc_inv[index_next] = rdc_inv[index]
        for f in fdc:
            for s in all_sku_list:
                index_del = gene_index(f, s, d)
                if fdc_forecast_sales.has_key(index_del):
                    fdc_forecast_sales[index_del]
                if fdc_forecast_sales.has_key(index_del):
                    fdc_forecast_std[index_del]

        start_date = datetime.datetime.strptime(d, '%Y-%m-%d') + datetime.timedelta(1)
        start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')
        if start_date not in date_range:
            continue
        tmp_sku_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sku/' + start_date + '.pkl'
        pkl_sale = open(tmp_sku_path)
        sku_day_data = pickle.load(pkl_sale)
        pkl_sale.close()
        tmp_fdc_forecast_sales = pd.concat(
            [sku_day_data['date_s'].astype('str') + sku_day_data['dc_id'].astype('str')
             + sku_day_data['sku_id'].astype('str'),
             sku_day_data['forecast_daily_override_sales']], axis=1)
        tmp_fdc_forecast_sales.columns = ['id', 'forecast_value']
        tmp_fdc_forecast_sales = tmp_fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()
        fdc_forecast_sales.update(copy.deepcopy(tmp_fdc_forecast_sales))
        tmp_fdc_forecast_std = pd.concat([sku_day_data['date_s'].astype('str') + sku_day_data['dc_id'].astype('str')
                                          + sku_day_data['sku_id'].astype('str'), sku_day_data['std']], axis=1)
        tmp_fdc_forecast_std.columns = ['id', 'forecast_std']
        tmp_fdc_forecast_std = tmp_fdc_forecast_std.set_index('id')['forecast_std'].to_dict()
        fdc_forecast_std.update(copy.deepcopy(tmp_fdc_forecast_std))


        tmp_df = sku_day_data[sku_day_data['white_flag'] == 1][['date_s', 'sku_id', 'dc_id']]
        for k, v in tmp_df['sku_id'].groupby([tmp_df['date_s'], tmp_df['dc_id']]):
            white_list_dict[k[1]][k[0]] = list(v)


