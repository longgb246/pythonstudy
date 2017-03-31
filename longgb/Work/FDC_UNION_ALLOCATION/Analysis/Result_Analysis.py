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
    # # kpi_need_fdc = sim_all_sku_retail_keep
    # kpi_need_fdc.columns
    # rdc_inv
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


def calKpi(kpi_need_fdc, suffix=''):
    '''
    需要字段：sku_id, fdc_id, inv_his, inv_sim, sales_sim, sales_his_origin
    '''
    # # kpi_need_fdc = sim_all_sku_retail_keep
    # kpi_need_fdc.columns
    # rdc_inv
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
        fdc_kpi['cr_his_new_0'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_0'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 0)) / float(30 * sku_cnt)
        fdc_kpi['cr_his_new_12'][tmp_fdcid] = sum((fdcdata.inv_his > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
        fdc_kpi['cr_sim_new_12'][tmp_fdcid] = sum((fdcdata.inv_sim > 0) | (fdcdata.rdc_inv > 12)) / float(30 * sku_cnt)
    sim_fdc_kpi=pd.DataFrame(fdc_kpi)
    sim_fdc_kpi.columns = map(lambda x: x + suffix,list(sim_fdc_kpi.columns))
    sim_fdc_kpi.reset_index(inplace=True)
    sim_fdc_kpi.rename(columns={'index': 'fdc_id'}, inplace=True)
    return sim_fdc_kpi


import matplotlib.pyplot as plt
# 2.2 主函数
def plotBoxPlot(data, size=(8, 8), diff_color=False, xlabeln='x', ylabeln='y', titlen='', xticklabels=[]):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    # boxplot的属性
    boxprops = dict(linewidth=2, facecolor='#4C72B0', alpha=0.35)           # 盒子属性
    whiskerprops = dict(linewidth=2.5, linestyle='--', color='#797979', alpha=0.8)          # 虚线条属性
    flierprops = dict(linewidth=2, marker='o', markerfacecolor='none', markersize=6, linestyle='none')  # 异常值
    medianprops = dict(linestyle='-', linewidth=2.5, color='#FFA455')       # 中位数
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='#C44E52')   # 均值
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='r', alpha=0.6)               # 均值
    capprops = dict(linestyle='-', linewidth=2.5, color='#797979', alpha=0.8)               # 边界横线
    bplot = ax.boxplot(data,
               vert=True,               # vertical box aligmnent
               showmeans=True,          # 显示均值
               meanline=True,           # 均值使用线
               patch_artist=True,       # fill with color
               boxprops=boxprops,       # 盒子属性
               whiskerprops=whiskerprops,   # 虚线条属性
               capprops=capprops,           # 边界横线
               flierprops=flierprops,       # 异常值
               medianprops=medianprops,     # 中位数  #FFA455   #797979    #3E3E3E
               meanprops=meanlineprops      # 异常值
                )
    colors = ['pink', 'lightblue', 'lightgreen', '#6AB27B', '#a27712', '#8172B2', '#4C72B0', '#C44E52', '#FFA455', '#797979'] * 4
    # 添加 box 的颜色
    if diff_color:
        for patch, color in zip(bplot['boxes'], colors[:len(bplot['boxes'])]):
            patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data))], )
    if xticklabels == []:
        xticklabels = ['x{0}'.format(x) for x in range(1, len(data)+1)]
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabeln)
    ax.set_ylabel(ylabeln)
    ax.set_title(titlen)
    return [fig, ax]


def plotHistPer_this(plot_data, binsn=[], xlabeln='x', ylabeln='y', titlen='', save_path='', cum_True=True, size=(12,8), is_int=True, is_save=False, is_drop_zero=False):
    '''
    画hist的百分比图，指定bins
    :param data: pd.DataFrame 单列数据
    :param binsn: numeric 指定的bins
    :param xlabeln: unicode x轴名称
    :param ylabeln: unicode y轴名称
    :param titlen: unicode 图的标题
    :param save_path: string 文件路径
    :param cum_True: boolean 是否添加累计概率线
    :param size: tuple 画图的size大小
    :param is_int: boolean 是否把标签变成整数
    :return: None 仅用于作图
    '''
    # plot_data=z_value_frame.z_value; binsn=[-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf]
    # xlabeln = u'z值'; ylabeln = u'频数'; titlen = u"Z值分布图"; size=(12,8); intshu=True
    plt.style.use('seaborn-darkgrid')
    if binsn == []:
        ret = plt.hist(plot_data, label='Z', color='#0070C0', histtype='bar', rwidth=0.6)
    else:
        ret = plt.hist(plot_data, bins=binsn, label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    plt.close()
    counts, bins, patches = ret[0], ret[1], ret[2]
    if is_int:
        bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x,bins)
    bins_name = ["["+str(bins[i])+","+str(bins[i+1])+")" for i in range(len(bins)-1)]
    bins_name[0] = bins_name[0].replace('[','(')
    bins_name = ['0'] + bins_name
    counts = list(counts)
    plot_data_count0 = np.sum(plot_data==0)
    counts = [plot_data_count0] + counts
    counts[1] = counts[1] - counts[0]
    counts = np.array(counts)
    if is_drop_zero:
        tmp_counts = []
        tmp_bins_name = []
        for i, each in enumerate(counts):
            if each != 0:
                tmp_counts.append(counts[i])
                tmp_bins_name.append(bins_name[i])
        counts = tmp_counts
        bins_name = tmp_bins_name
    ind = np.arange(len(counts))
    fig1, ax1 = plt.subplots(figsize=size)
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    width = 0.5
    width2 = 0
    ax1.bar(ind + width2, counts, width, color="#0070C0", tick_label=bins_name, align='center', alpha=0.6)
    counts_per = counts/np.sum(counts)
    counts_per_cum = np.cumsum(counts_per)
    i = 0
    ymin, ymax = plt.ylim()
    ax1.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
    # ax1.set_xlim(-1, len(bins_name)+1)
    for x, y in zip(ind, counts):
        ax1.text(x + width2, y + 0.05, '{0:.2f}%'.format(counts_per[i]*100), ha='center', va='bottom')
        i += 1
    plt.title(titlen)
    if cum_True:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative probability distribution')
        ax2.plot(ind + width2, counts_per_cum, '--', color="red")
        ax2.yaxis.grid(False)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(-0.5, len(bins_name) - 0.5)
    plt.show()
    if is_save:
        plt.savefig(save_path)
    return [fig1, ax1, ax2, bins_name, counts]


# ======================================================================
# =                                 计算函数                           =
# ======================================================================
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

    # 1、计算分段的总量
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


def allocationQttyDiff():
    skulist = pd.read_csv(read_path + os.sep + 'skulist.csv')
    skulist_need = skulist[skulist['countalloctionreal'] > 0]

    sim_all_sku_retail = pd.read_table(read_path + os.sep + 'sim_all_sku_retail.csv')
    system_all_sku_retail = pd.read_table(read_path + os.sep + 'system_all_sku_retail.csv')
    sim_all_sku_retail['allocation_retail_diff'] = sim_all_sku_retail['allocation_retail_cacl'] - sim_all_sku_retail['allocation_retail_real']
    system_all_sku_retail['allocation_retail_diff'] = system_all_sku_retail['allocation_retail_cacl'] - system_all_sku_retail['allocation_retail_real']

    # 1、存储明细信息1
    sim_all_sku_retail_save = skulist_need.merge(sim_all_sku_retail, on=['fdc_id','sku_id'])
    sim_all_sku_retail_save = sim_all_sku_retail_save.loc[:,['dt', 'fdc_id', 'sku_id', 'allocation_retail_cacl', 'allocation_retail_real', 'allocation_retail_diff']]
    system_all_sku_retail_save = skulist_need.merge(system_all_sku_retail, on=['fdc_id','sku_id'])
    system_all_sku_retail_save = system_all_sku_retail_save.loc[:,['dt', 'fdc_id', 'sku_id', 'allocation_retail_cacl', 'allocation_retail_real', 'allocation_retail_diff']]
    sim_all_sku_retail_save.to_csv(read_path + os.sep + 'allocationQttyDiff_sim_detail.csv', index=False)
    system_all_sku_retail_save.to_csv(read_path + os.sep + 'allocationQttyDiff_system_detail.csv', index=False)

    # 仿真
    sim_all_sku_retail_keep = pd.DataFrame()
    for key, value in sim_all_sku_retail.groupby(['fdc_id','sku_id']):
        tmp_pd = pd.DataFrame.from_dict({'fdc_id':key[0],'sku_id':key[1],'diff_sum':np.sum(value['allocation_retail_diff'])}, orient='index').T
        sim_all_sku_retail_keep = pd.concat([sim_all_sku_retail_keep, tmp_pd])
    sim_all_sku_retail_keep.index = range(len(sim_all_sku_retail_keep))

    diff_pd = skulist_need.merge(sim_all_sku_retail_keep, on=['fdc_id','sku_id'])
    fdc_list = diff_pd['fdc_id'].unique()
    count_table = pd.DataFrame()
    for fdc_id in fdc_list:
        diff_pd_fdc = diff_pd[diff_pd['fdc_id']==fdc_id]
        binsn = [0, 10, 20, 30, 40, 50, 100, 200, 300, np.inf]
        fig1, ax1, ax2, bins_name, counts = plotHistPer_this(diff_pd_fdc['diff_sum'], binsn=binsn)
        tmp_pd = pd.DataFrame(np.matrix([fdc_id] + list(counts) + [np.sum(counts)]), columns=['fdc_id'] + bins_name + ['Total'])
        count_table = pd.concat([count_table, tmp_pd])
    count_table.to_csv(read_path + os.sep + 'allocationQttyDiff_sim.csv', index=False)

    # 系统
    system_all_sku_retail_keep = pd.DataFrame()
    for key, value in system_all_sku_retail.groupby(['fdc_id','sku_id']):
        tmp_pd = pd.DataFrame.from_dict({'fdc_id':key[0],'sku_id':key[1],'diff_sum':np.sum(value['allocation_retail_diff'])}, orient='index').T
        system_all_sku_retail_keep = pd.concat([system_all_sku_retail_keep, tmp_pd])
    system_all_sku_retail_keep.index = range(len(system_all_sku_retail_keep))

    diff_pd_system = skulist_need.merge(system_all_sku_retail_keep, on=['fdc_id', 'sku_id'])
    fdc_list = diff_pd_system['fdc_id'].unique()
    count_table = pd.DataFrame()
    for fdc_id in fdc_list:
        diff_pd_fdc = diff_pd_system[diff_pd_system['fdc_id'] == fdc_id]
        binsn = [0, 10, 20, 30, 40, 50, 100, 200, 300, np.inf]
        fig1, ax1, ax2, bins_name, counts = plotHistPer_this(diff_pd_fdc['diff_sum'], binsn=binsn)
        tmp_pd = pd.DataFrame(np.matrix([fdc_id] + list(counts) + [np.sum(counts)]), columns=['fdc_id'] + bins_name + ['Total'])
        count_table = pd.concat([count_table, tmp_pd])
    count_table.to_csv(read_path + os.sep + 'allocationQttyDiff_system.csv', index=False)

    # 2、存储明细信息2
    diff_pd_save = diff_pd.loc[:,['fdc_id','sku_id','diff_sum']]
    diff_pd_system_save = diff_pd_system.loc[:,['fdc_id','sku_id','diff_sum']]
    diff_pd_save.to_csv(read_path + os.sep + 'allocationQttyDiff_sim_detail2.csv', index=False)
    diff_pd_system_save.to_csv(read_path + os.sep + 'allocationQttyDiff_system_detail2.csv', index=False)


def kpi_analysis():
    analysis_path = r'D:\Lgb\WorkFiles\FDC_UNION_ALLOCATION\analysis_3_policy'
    files = ['simulation_results_L_std', 'simulation_results_S', 'simulation_results_select']
    suffix = ['_L_std', '_S', '_8']
    band = pd.read_table(analysis_path + os.sep + 'SKUABCband_20170330153011.csv')
    band.columns = ['sku_id', 'label']
    first = 1
    kpi_list = []
    kpi_band_list = []
    for i, each_file in enumerate(files):
        # 1、--------------------------- 读取文件 ---------------------------
        sim_all_sku_kpi = pd.read_table(analysis_path + os.sep + each_file + os.sep + 'sim_all_sku_kpi.csv')
        system_all_sku_kpi = pd.read_table(analysis_path + os.sep + each_file + os.sep + 'system_all_sku_kpi.csv')
        sim_all_sku_retail = pd.read_table(analysis_path + os.sep + each_file + os.sep + 'sim_all_sku_retail.csv')
        system_all_sku_retail = pd.read_table(analysis_path + os.sep + each_file + os.sep + 'system_all_sku_retail.csv')
        sim_all_sku_retail = sim_all_sku_retail.merge(band, on=['sku_id'])
        system_all_sku_retail = system_all_sku_retail.merge(band, on=['sku_id'])
        if first == 0:
            sku_del_list = []
            # 2、--------------------------- 筛选 剔除总销量 为0的 ---------------------------
            for key, value in sim_all_sku_retail.groupby(['fdc_id', 'sku_id']):
                if np.sum(value['sales_his_origin']) == 0:
                    sku_del_list.append(key)
            first += 1
            sku_all_list = map(lambda x: (x[0], x[1]),sim_all_sku_retail.loc[:, ['fdc_id', 'sku_id']].drop_duplicates().values)
            sku_keep_list = set(sku_all_list) - set(sku_del_list)
            sku_del_list_pd = pd.DataFrame(sku_del_list, columns=['fdc_id', 'sku_id'])
            sku_del_list_pd.to_csv(analysis_path + os.sep + 'sku_del_list.csv', index=False)
            sku_keep_list_pd = pd.DataFrame(list(sku_keep_list), columns=['fdc_id', 'sku_id'])
            sku_keep_list_pd.to_csv(analysis_path + os.sep + 'sku_keep_list.csv', index=False)
        else:
            sku_del_list_pd = pd.read_csv(analysis_path + os.sep + 'sku_del_list.csv')
            sku_keep_list_pd = pd.read_csv(analysis_path + os.sep + 'sku_keep_list.csv')
        sim_all_sku_retail_keep = sim_all_sku_retail.merge(sku_keep_list_pd, on=['fdc_id', 'sku_id'])
        system_all_sku_retail_keep = system_all_sku_retail.merge(sku_keep_list_pd, on=['fdc_id', 'sku_id'])
        # 3、--------------------------- 计算 KPI ---------------------------
        sim_all_sku_retail_keep_kpi = calKpi(sim_all_sku_retail_keep, suffix='_sim' + suffix[i])
        system_all_sku_retail_keep_kpi = calKpi(system_all_sku_retail_keep, suffix='_system' + suffix[i])
        kpi_list.append(sim_all_sku_retail_keep_kpi)
        kpi_list.append(system_all_sku_retail_keep_kpi)
        sim_all_sku_retail_keep_kpi_band = calKpi_label(sim_all_sku_retail_keep, suffix='_sim' + suffix[i])
        system_all_sku_retail_keep_kpi_band = calKpi_label(system_all_sku_retail_keep, suffix='_system' + suffix[i])
        kpi_band_list.append(sim_all_sku_retail_keep_kpi_band)
        kpi_band_list.append(system_all_sku_retail_keep_kpi_band)
    # 4、--------------------------- 合并 KPI ---------------------------
    kpi_list_1 = kpi_list[0].merge(kpi_list[1], on=['fdc_id']).merge(kpi_list[2], on=['fdc_id']).merge(kpi_list[4], on=['fdc_id'])

    sorted(kpi_list_1.columns)

    kpi_list_2 = kpi_band_list[0].merge(kpi_band_list[1], on=['fdc_id', 'label']).merge(kpi_band_list[2], on=['fdc_id', 'label']).merge(kpi_band_list[4], on=['fdc_id', 'label'])
    kpi_list_1_keep = kpi_list_1.loc[:,['fdc_id',
                                        'cr_his_sim_8', 'cr_sim_sim_L_std', 'cr_sim_sim_S', 'cr_sim_sim_8', 'cr_sim_system_L_std',
                                        'ito_his_sim_8', 'ito_sim_sim_L_std', 'ito_sim_sim_S', 'ito_sim_sim_8', 'ito_sim_system_L_std',
                                        'ts_his_sim_8', 'ts_sim_sim_L_std', 'ts_sim_sim_S', 'ts_sim_sim_8', 'ts_sim_system_L_std',
                                        'ts_rate_sim_L_std', 'ts_rate_sim_S', 'ts_rate_sim_8', 'ts_rate_system_L_std']]
    kpi_list_2_keep = kpi_list_2.loc[:,['fdc_id', 'label',
                                        'cr_his_sim_8', 'cr_sim_sim_L_std', 'cr_sim_sim_S', 'cr_sim_sim_8', 'cr_sim_system_L_std',
                                        'ito_his_sim_8', 'ito_sim_sim_L_std', 'ito_sim_sim_S', 'ito_sim_sim_8', 'ito_sim_system_L_std',
                                        'ts_his_sim_8', 'ts_sim_sim_L_std', 'ts_sim_sim_S', 'ts_sim_sim_8', 'ts_sim_system_L_std',
                                        'ts_rate_sim_L_std', 'ts_rate_sim_S', 'ts_rate_sim_8', 'ts_rate_system_L_std']]
    # 5、--------------------------- 存储 KPI ---------------------------
    kpi_list_1_keep.to_csv(analysis_path + os.sep + 'kpi_list_1_keep.csv', index=False)
    kpi_list_2_keep.to_csv(analysis_path + os.sep + 'kpi_list_2_keep.csv', index=False)
    # 6、new 分析
    kpi_list_3_keep = kpi_list_1.loc[:,['fdc_id',
                                        'cr_his_sim_8', 'cr_sim_sim_L_std', 'cr_sim_sim_S', 'cr_sim_sim_8', 'cr_sim_system_L_std',
                                        'ito_his_sim_8', 'ito_sim_sim_L_std', 'ito_sim_sim_S', 'ito_sim_sim_8', 'ito_sim_system_L_std',
                                        'ts_his_sim_8', 'ts_sim_sim_L_std', 'ts_sim_sim_S', 'ts_sim_sim_8', 'ts_sim_system_L_std',
                                        'ts_rate_sim_L_std', 'ts_rate_sim_S', 'ts_rate_sim_8', 'ts_rate_system_L_std',
                                        'cr_his_new_0_sim_L_std', 'cr_sim_new_0_sim_L_std', 'cr_his_new_12_sim_L_std', 'cr_sim_new_12_sim_L_std']]
    kpi_list_4_keep = kpi_list_2.loc[:,['fdc_id', 'label',
                                        'cr_his_sim_8', 'cr_sim_sim_L_std', 'cr_sim_sim_S', 'cr_sim_sim_8', 'cr_sim_system_L_std',
                                        'ito_his_sim_8', 'ito_sim_sim_L_std', 'ito_sim_sim_S', 'ito_sim_sim_8', 'ito_sim_system_L_std',
                                        'ts_his_sim_8', 'ts_sim_sim_L_std', 'ts_sim_sim_S', 'ts_sim_sim_8', 'ts_sim_system_L_std',
                                        'ts_rate_sim_L_std', 'ts_rate_sim_S', 'ts_rate_sim_8', 'ts_rate_system_L_std',
                                        'cr_his_new_0_sim_L_std', 'cr_sim_new_0_sim_L_std', 'cr_his_new_12_sim_L_std', 'cr_sim_new_12_sim_L_std']]
    kpi_list_3_keep.to_csv(analysis_path + os.sep + 'kpi_list_3_keep.csv', index=False)
    sorted(kpi_list_3_keep.columns)
    kpi_list_4_keep.to_csv(analysis_path + os.sep + 'kpi_list_4_keep.csv', index=False)


def kpi_cr_new():

    pass


if __name__ == '__main__':
    pass

