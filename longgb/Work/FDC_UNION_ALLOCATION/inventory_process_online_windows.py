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
    def getCategory(sales, ):
        category = 99  # normal
        percent = (sum(sales > 0) * 1.0 / len(sales))
        salesMean = np.mean(sales)
        if (percent >= category_longTail_stable_DaysThreshold) & (salesMean <= category_longTail_stable_SalesThreshold):
            category = 1  # longTail_stable
        return category

    @staticmethod
    def getPredictionErrorMultiple(sales, pred_sales, cur_index):
        """
        judge whether prediction sales exceed the actual sales
        """
        sales3days = sum([sales[cur_index]] * 3)
        pred_sales3days = sum([pred_sales[cur_index][0]] * 3)
        if cur_index >= 3:
            sales3days = sum(sales[cur_index - 3:cur_index])
            pred_sales3days = sum(pred_sales[cur_index - 3][0:3])
        multiple = max((sales3days * 1.0 / pred_sales3days), 1)
        return multiple

    @staticmethod
    def getWeightedActSales(sales, cur_index):
        """
        1. estimate whether error is too large
        2. return weighted
        """
        if cur_index >= salespredictionErrorFilldays:
            actualSale = sales[cur_index - salespredictionErrorFilldays:cur_index]
            return [np.mean(actualSale)], [np.std(actualSale)]
        else:
            rang = salespredictionErrorFilldays - cur_index
            mean_sale = np.nanmean(sales[0:cur_index])
            actualSale = np.concatenate((sales[0:cur_index], np.array([mean_sale] * (rang))))
            return [np.mean(actualSale)], [np.std(actualSale)]


# ---------------------------- 不需要重新运行 ----------------------------
fdc_forecast_sales = 1
fdc_forecast_std = 1
fdc_alt = 1
fdc_alt_prob = 1
fdc_inv = 1
sku_id = 1
fdc_list = 1
rdc_inv = 1
order_list = 1
date_range = 1
logger = 1
save_data_path = 1
sales_retail = 1
white_flag = 1
fdc_his_inv = 1
rdc_sale_list = 1
system_small_s = 1
system_bigger_S = 1


# ---------------------------- 需要重新运行 ------------------------------
sku = sku_id
fdc_allocation = defaultdict(float)
allocation_retail = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
fdc_sales_retail = defaultdict(int)
sim_sales_retail = defaultdict(int)
lop = defaultdict(int)
alt_sim = defaultdict(int)
white_list_dict = white_flag
all_cnt_sim = defaultdict(int)                  # 记录FDC调拨次数
system_flag = 0
fdc_begin_inv = defaultdict(int)


def gene_whitelist(date_s):
    # date_s = d
    '''获取该时间点的白名单,调用户一次刷新一次，只保存最新的白名单列表'''
    white_list = defaultdict(int)
    for f in fdc_list:
        for k, v in white_list_dict.items():
            if k == date_s:
                white_list[f] = v[f]


def cacl_rdc_inv(date_s):
    # date_s = d
    '''  补货逻辑 #更新RDC库存,RDC库存的更新按照实际订单情况进行更新，rdc{index:库存量},同时扣除rdc销量'''
    index = gene_index('rdc', sku, date_s)
    rdc_inv[index] = rdc_inv[index] + order_list[date_s].get(sku, 0)
    rdc_inv[index] = max(rdc_inv[index] - rdc_sale_list[date_s].get(sku, 0), 0)


def calc_lop(fdc, date_s, cr=0.99):
    '''    #计算某个FDC的某个SKU的补货点'''
    # fdc = f
    index = gene_index(fdc, sku, date_s)
    sku_sales = fdc_forecast_sales[index]
    try:
        sku_std = fdc_forecast_std[index]
    except:
        sku_std = 0
    sku_sales_mean = np.mean(sku_sales)
    if system_flag == 1:
        lop_1 = sku_sales_mean * system_small_s[index]
    else:
        lop_1 = sku_sales[0] + sku_sales[1] + sku_sales[2] + 1.96 * sku_std
    lop[index] = lop_1
    return lop


def calc_replacement(fdc, date_s, sku_lop, bp=10, cr=0.99):
    '''
    #计算某个FDC的SKU的补货量
    计算补货量补货量为lop+bp-在途-当前库存
    '''
    # sku的FDC销量预测，与RDC的cv系数
    index = gene_index(fdc, sku, date_s)
    sku_sales = fdc_forecast_sales[index]
    try:
        sku_std = fdc_forecast_std[index]
    except:
        sku_std = 0
    sku_sales_mean = np.mean(sku_sales)
    max_qtty = sku_sales_mean * system_bigger_S[index]
    inv = fdc_inv[index]['inv']
    open_on = fdc_inv[index]['open_po']
    if system_flag == 1:
        lop_replacement = max(max_qtty - inv - open_on, 0)                  # 【标记】
    else:
        lop_replacement = sku_sales_mean * 7
    # 调整补货量
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


def calc_sku_allocation(date_s):
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
    fdc_replacement = defaultdict(int)
    for f in fdc_list:
        # f = fdc_list[0]
        if white_list[f] == 0:
            fdc_replacement[f] = 0
        else:
            lop_tmp = calc_lop(f, date_s)
            index = gene_index(f, sku, date_s)
            # 在途加上库存减去在途消耗低于补货点
            if (fdc_inv[index]['inv'] + fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po']) < lop_tmp:
                fdc_replacement[f] = calc_replacement(f, date_s, lop_tmp)
                all_cnt_sim[f] = all_cnt_sim[f] + 1
            else:
                fdc_replacement[f] = 0
    index_rdc = gene_index('rdc', sku, date_s)
    rdc_inv_avail = max(np.min([rdc_inv[index_rdc] - 12, np.floor(rdc_inv[index_rdc] * 0.2)]), 0)
    # 记录FDC的库存
    allocation_retail[date_s][sku]['rdc'] = rdc_inv[index_rdc]
    # 更新实际调拨，记录理论调拨和实际调拨，之所以将
    for f in fdc_list:
        index_fdc = gene_index(f, sku, date_s)
        allocation_retail[date_s][sku]['calc_' + str(f)] = fdc_replacement[f]
        fdc_inv_avail = np.min([fdc_replacement[f], rdc_inv_avail])
        allocation_retail[date_s][sku]['real_' + str(f)] = fdc_inv_avail
        fdc_allocation[index_fdc] = fdc_inv_avail
        rdc_inv[index_rdc] = rdc_inv[index_rdc] - fdc_inv_avail
        rdc_inv_avail -= fdc_inv_avail


def gene_index(fdc, sku, date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s) + ':' + str(fdc) + ':' + str(sku)


def gene_alt(fdc):
    '''
    生成对应的调拨时长，用以更新FDC的库存
    '''
    fdc_vlt = fdc_alt[fdc]
    fdc_vlt_porb = fdc_alt_prob[fdc]
    # 如果没有对应的调拨时长，默认为3天
    if len(fdc_vlt) == 0:
        return 3
    alt_distribution = rv_discrete(values=(fdc_vlt, fdc_vlt_porb))
    return alt_distribution.rvs()


def calc_fdc_allocation(date_s, fdc):
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
    index = gene_index(fdc, sku, date_s)
    # 获取当前库存，当前库存已在订单循环部分完成
    # 获取调拨量,从调拨字典中获取调拨量
    if fdc_allocation[index] > 0:
        fdc_inv[index]['allocation'] = fdc_allocation[index]
        # 放在这里，因为同一个调拨单的也不能保证同一天到达，所以按照SKU进行时长抽样
        alt = -1
        while (alt < 1 or alt > 5):
            alt = gene_alt(fdc)
        # 保持alt信息
        alt_sim[index] = alt
        # 更新在途量,c为标记变量
        c = 0
        format_date = '%Y-%m-%d'
        while c < alt:
            date_tmp = datetime.datetime.strptime(date_s, format_date) + datetime.timedelta(c)
            date_s_c = date_tmp.strftime('%Y-%m-%d')
            index_tmp = gene_index(fdc, sku, date_s_c)
            fdc_inv[index_tmp]['open_po'] = fdc_inv[index]['allocation'] + fdc_inv[index_tmp]['open_po']
            c += 1
        date_alt = datetime.datetime.strptime(date_s, format_date) + datetime.timedelta(alt)
        date_s_alt = date_alt.strftime(format_date)
        index_1 = gene_index(fdc, sku, date_s_alt)
        # 更新当日到达量
        fdc_inv[index_1]['arrive_quantity'] = fdc_inv[index]['allocation'] + fdc_inv[index_1]['arrive_quantity']
    # 当天库存等于 当天00:00:00的库存+当天到达-在途消耗，当天到达量可能低于在途消耗，所以需要预处理
    # 当天实际到达量为 min(fdc_inv[index]['arrive_quantity']-fdc_inv[index]['cons_open_po'],0),同时更新在途消耗为
    # max(fdc_inv[index]['cons_open_po']-fdc_inv[index]['arrive_quantity']，0),在途消耗仅仅小于在途量，与其他变量无直接关系
    fdc_inv[index]['inv'] = fdc_inv[index]['inv'] + \
                            max(fdc_inv[index]['arrive_quantity'] - fdc_inv[index]['cons_open_po'], 0)
    fdc_inv[index]['cons_open_po'] = max(fdc_inv[index]['cons_open_po'] - fdc_inv[index]['arrive_quantity'], 0)


def gene_fdc_datasets():
    sku_allocation = defaultdict(int)
    sku_open_po = defaultdict(int)
    sku_arrive_quantity = defaultdict(int)
    sku_inv = defaultdict(int)
    sku_cons_open_po = defaultdict(int)
    for k, v in fdc_inv.items():
        sku_allocation[k] = v['allocation']
        sku_open_po[k] = v['open_po']
        sku_arrive_quantity[k] = v['arrive_quantity']
        sku_inv[k] = v['inv']
        sku_cons_open_po[k] = v['cons_open_po']
    allocation_retail_real = defaultdict(int)
    allocation_retail_cacl = defaultdict(int)
    allocation_retail_rdc = defaultdict(int)
    for k_date, v in allocation_retail.items():
        for k_sku, v1 in v.items():
            for k_fdc, v2 in v1.items():
                if 'rdc' not in k_fdc:
                    allocation_value_type = k_fdc.split('_')[0]
                    allocation_fdc = k_fdc.split('_')[1]
                    if allocation_value_type == 'real':
                        tmp_index = gene_index(allocation_fdc, k_sku, k_date)
                        allocation_retail_real[tmp_index] = v2
                    elif allocation_value_type == 'calc':
                        tmp_index = gene_index(allocation_fdc, k_sku, k_date)
                        allocation_retail_cacl[tmp_index] = v2
                else:
                    tmp_index = gene_index('rdc', k_sku, k_date)
                    allocation_retail_rdc[tmp_index] = v2


def get_daily_data():
    gene_fdc_datasets()
    # print sku_inv
    # print fdc_his_inv
    daily_data = {'sales_his_origin': sales_retail,
                  'inv_his': fdc_his_inv,
                  'inv_sim': fdc_begin_inv,
                  'cons_open_po': sku_cons_open_po,
                  'sales_sim': sim_sales_retail,
                  'fdc_sales_sim': fdc_sales_retail,
                  'lop': lop,
                  'allocation_qtty_sim': sku_allocation,
                  'open_po_sim': sku_open_po,
                  'alt_sim': alt_sim,
                  'arrive_qtty_sim': sku_arrive_quantity,
                  'allocation_retail_real': allocation_retail_real,
                  'allocation_retail_cacl': allocation_retail_cacl,
                  'rdc_inv': rdc_inv}
    daily_data_mid = pd.DataFrame(daily_data, columns=['sales_his_origin', 'inv_his', 'inv_sim',
                                                       'cons_open_po', 'sales_sim', 'fdc_sales_sim', 'lop',
                                                       'allocation_qtty_sim', 'open_po_sim', 'alt_sim',
                                                       'arrive_qtty_sim',
                                                       'allocation_retail_real', 'allocation_retail_cacl', 'rdc_inv'])
    daily_data_mid.fillna(0, inplace=True)
    daily_data_mid.reset_index(inplace=True)
    daily_data_mid2 = pd.DataFrame(list(daily_data_mid['index'].apply(lambda x: x.split(':'), 1)))
    daily_data_mid2.columns = ['dt', 'fdc_id', 'sku_id']
    # print 'fdc_id',pd.unique(daily_data_mid2.fdc_id)
    del daily_data_mid['index']
    reuslt_daily_data = daily_data_mid2.join(daily_data_mid)
    return reuslt_daily_data[reuslt_daily_data['fdc_id'] != 'rdc']


def calc_kpi():
    fdc_kpi = defaultdict(lambda: defaultdict(float))
    for tmp_fdcid, fdcdata in reuslt_daily_data.groupby(['fdc_id']):
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
    tmp_mid_kpi = pd.DataFrame(fdc_kpi)
    tmp_mid_kpi.reset_index(inplace=True)
    tmp_mid_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    tmp_mid_kpi['sku_id'] = sku
    # print tmp_mid_kpi
    return tmp_mid_kpi


def allocationSimulation():
    rdc_fdc = fdc_list
    for d in date_range:
        # 1.1 更新获取当天白名单
        gene_whitelist(d)
        # 1.2 更新RDC库存，增
        cacl_rdc_inv(d)
        # 1.3 计算每个SKU的调拨量
        calc_sku_allocation(d)
        # 更新FDC的调拨量
        for f in rdc_fdc:
            # 更新FDC当天库存，增
            # 并针对FDC进行调拨，仅更新调拨、在途、到达，并未减
            calc_fdc_allocation(d, f)
            # ----------------------更新SKU的库存信息------------------------------------------------------------------------------------------#
            index = gene_index(f, sku, d)
            rdc_index = gene_index('rdc', sku, d)
            ###标记初始化库存
            # print index,fdc_inv[index]['inv']
            fdc_begin_inv[index] = fdc_inv[index]['inv']
            ###标记完成
            # FDC销量标记与总销量标记放在第一位，因为涉及到在途消耗，该尚未造成实际的消耗增加
            fdc_sales_retail[index] = min(sales_retail[index],
                                          fdc_inv[index]['inv'] + fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po'])
            sim_sales_retail[index] = min(sales_retail[index],
                                          fdc_inv[index]['inv'] + fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po'] + rdc_inv[rdc_index])      # 【记录】
            # 记录消耗在途的数量
            if sales_retail[index] > fdc_inv[index]['inv']:
                # 该递推公式保证了 fdc_inv[index]['cons_open_po'] 小于等于  fdc_inv[index]['open_po']
                # 同时 在途消耗为正值，s[1]>fdc_inv[index]['inv'] 保证了 第一项>0
                fdc_inv[index]['cons_open_po'] = fdc_inv[index]['cons_open_po'] + min(sales_retail[index] - fdc_inv[index]['inv'],
                    fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po'])
            # 库存放在最好此时的标记订单已经消耗实际库存
            # 首先从fdc 发货 其次不够的从rdc补，需求>=库存，则fdc库存为0,否则为 剩余量
            sku_gap = sales_retail[index] - fdc_inv[index]['inv']
            fdc_inv[index]['inv'] = 0 if sku_gap >= 0 else abs(sku_gap)
            # 在模拟中有些订单会不被满足，所以需要在0 和实际值之间取最大值，无效订单在simu_order里面会被标记
            sku_gap = sales_retail[index] - (fdc_inv[index]['inv'] + fdc_inv[index]['open_po'] - fdc_inv[index]['cons_open_po'])
            rdc_inv[rdc_index] = rdc_inv[rdc_index] if sku_gap < 0 else max(rdc_inv[rdc_index] - sku_gap, 0)                                            # rdc 保留量的考虑？
            # ----------------------------------------------------------------------------------------------------------------#
        # 更新下一天库存，将当天剩余库存标记为第二天库存,第二天到达库存会在开始增加上，将每天最后的在途消耗 更新为第二天的初始在途消耗，在第二天更新调拨的时候
        # 在途消耗与第二天的到达量做运算，如果有到达，则在途消耗做减法运算，即 不入库直接发给用户
        for f in fdc_list:
            format_date = '%Y-%m-%d'
            date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
            date_s_c = date_tmp.strftime('%Y-%m-%d')
            index_next = gene_index(f, sku, date_s_c)
            index = gene_index(f, sku, d)
            fdc_inv[index_next]['inv'] = fdc_inv[index]['inv']
            fdc_inv[index_next]['cons_open_po'] = fdc_inv[index]['cons_open_po']
        # 不仅仅更新白名单更新全量库存，如何获取全量list
        format_date = '%Y-%m-%d'
        date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
        date_s_c = date_tmp.strftime('%Y-%m-%d')
        index_next = gene_index('rdc', sku, date_s_c)
        index = gene_index('rdc', sku, d)
        rdc_inv[index_next] = rdc_inv[index]
