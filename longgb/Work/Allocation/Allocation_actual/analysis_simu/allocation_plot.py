#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


def plothistper(plot_data, binsn, xlabeln, ylabeln, titlen, save_path, cum_True=True, size=(12, 8), intshu=True):
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
    :param intshu: boolean 是否把标签变成整数
    :return: None 仅用于作图
    '''
    # plot_data=Kpi_report_plot_fill; binsn=[-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf]
    # xlabeln = u'z值'; ylabeln = u'频数'; titlen = u"Z值分布图"; size=(12,8); intshu=True
    ret = plt.hist(plot_data, bins=binsn, label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    counts, bins, patches = ret[0], ret[1], ret[2]
    if intshu:
        bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x,bins)
    else:
        bins = map(lambda x: x if (np.abs(x - 0) > 0.00005) else 0,bins)
    counts_non = []
    bins_name = []
    for i, each in enumerate(counts):
        if each != 0:
            counts_non.append(each)
            bins_name.append("["+str(bins[i])+","+str(bins[i+1])+")")
    counts = counts_non
    # bins_name = ["["+str(bins[i])+","+str(bins[i+1])+")" for i in range(len(bins)-1)]
    ind = np.arange(len(counts))
    fig1, ax1 = plt.subplots(figsize=size)
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    width = 0.5
    width2 = 0
    ax1.bar(ind + width2, counts, width, color="#0070C0", tick_label=bins_name, align='center')
    counts_per = counts/np.sum(counts)
    counts_per_cum = np.cumsum(counts_per)
    i = 0
    ymin, ymax = plt.ylim()
    ax1.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
    for x, y in zip(ind, counts):
        ax1.text(x + width2, y + 0.05, '{0:.2f}%'.format(counts_per[i]*100), ha='center', va='bottom')
        i += 1
    plt.title(titlen)
    if cum_True:
        ax2 = ax1.twinx()
        ax2.set_ylabel(u'累计概率分布')
        ax2.plot(ind + width2, counts_per_cum, '--', color="red")
        ax2.yaxis.grid(False)
        ax2.set_ylim(-0.05, 1.05)
    plt.savefig(save_path)


def plotsku(sku_data, kpi, filePath):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_xlabel(u'日期')
    ax1.set_ylabel(u'库存')
    ax2.set_ylabel(u'销量')
    ax2.yaxis.grid(False)
    # ax1.set_ylim(0, sku_data.stock_qtty.max() * 1.5)
    ax2.set_ylim(0, sku_data.total_sales.max() * 1.5)
    # plt.title(u'sku:%d 周转:%.2f 现货率:%.2f' % (id, kpi.TD[id], kpi.CR[id]))
    ax1.plot(sku_data.day_string, sku_data.stock_qtty, '-', label=u'库存', color='#6AB27B') #6AB27B #E4D354
    ax2.plot(sku_data.day_string, sku_data.total_sales, '-', label=u'销量', color='#0070C0')#4C72B0 #91E8E1
    legend1 = ax1.legend(loc=(.89, .94), fontsize=12, shadow=True)
    legend2 = ax2.legend(loc=(.89, .9), fontsize=12, shadow=True)
    fname = filePath + '\\' + str(id)
    plt.savefig(fname)
    plt.close()


def plotinv(data):
    data = sku_report_tmp.sort(['date_s'])
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(u'日期')
    ax1.set_ylabel(u'库存')
    for each in ['628', '630', '658']:
        pass
    data.index = range(len(data))
    ax1.plot(data['date_s'], data['inv'], '-', label=u'库存', color='#6AB27B')  # 6AB27B #E4D354
    ax1.legend(loc=(.89, .94), fontsize=12, shadow=True)


def plot_kpi_fill(Kpi_report, save_name, size=(12, 8), step=0.01):
    # Kpi_report_plot = Kpi_report.loc[:,['fdc', 'date_s','satisfy_rate_actual','fill_rate_sim', 'ito_actual','ito_rate_sim']]
    Kpi_report_plot_fill = Kpi_report['fill_rate_sim'] - Kpi_report['satisfy_rate_actual']
    max_num = int(np.ceil(Kpi_report_plot_fill.max() * 100))*1.0 / 100
    min_num = int(np.floor(Kpi_report_plot_fill.min() * 100))*1.0 / 100
    binsn = np.arange(min_num,max_num,step)
    xlabeln = u'差值'
    ylabeln = u'天数'
    titlen = u'{0} （仿真满足率 - 实际满足率）差值图 '.format(save_name)
    intshu = True if isinstance(step, int) else False
    plothistper(Kpi_report_plot_fill, binsn, xlabeln, ylabeln, titlen, save_path + os.sep + 'Kpi_report_fill_rate' + save_name + '.png', intshu=intshu, size=size)


def plot_kpi_ito(Kpi_report, save_name, size=(18, 12), step=1):
    # Kpi_report_plot = Kpi_report.loc[:,['fdc', 'date_s','satisfy_rate_actual','fill_rate_sim', 'ito_actual','ito_rate_sim']]
    Kpi_report_plot_fill = Kpi_report['ito_rate_sim'] - Kpi_report['ito_actual']
    max_num = int(np.ceil(Kpi_report_plot_fill.max()))*1.0
    min_num = int(np.floor(Kpi_report_plot_fill.min()))*1.0
    binsn = np.arange(min_num,max_num,step)
    xlabeln = u'差值'
    ylabeln = u'天数'
    titlen = u'{0} （仿真周转 - 实际周转）差值图 '.format(save_name)
    intshu = True if isinstance(step, int) else False
    plothistper(Kpi_report_plot_fill, binsn, xlabeln, ylabeln, titlen, save_path + os.sep + 'Kpi_report_ito' + save_name + '.png', size=size, intshu=intshu)


# =======================================================================
# =                                 路径设置                            =
# =======================================================================
# 路径
report_path = r'D:\Lgb\ReadData\actual'
save_path = r'D:\Lgb\ReadData\actual'


if __name__ == '__main__':
    # ===============================================================================
    # =                                 （1）Kpi 数据表                             =
    # ===============================================================================
    # 1、读取数据
    Kpi_report = pd.read_csv(report_path + os.sep + 'Kpi_report.csv')
    Kpi_report['fdc'] = Kpi_report['fdc'].astype(str)
    type_sure = Kpi_report.columns[2:]
    for each in type_sure:
        Kpi_report[each] = Kpi_report[each].astype(float)

    for fdc in ['628', '630', '658']:
        Kpi_report_tmp = Kpi_report[Kpi_report['fdc'] == fdc]
        plot_kpi_fill(Kpi_report_tmp, fdc)
        plot_kpi_ito(Kpi_report_tmp, fdc, size=(12, 8), step=0.5)
    plot_kpi_fill(Kpi_report, 'Total', size=(18, 12), step=0.02)
    plot_kpi_ito(Kpi_report, 'Total', size=(18, 12), step=1)

    # sku_report = pd.read_csv(report_path + os.sep + 'table_sample_sku_sample_select.csv')
    # sku_report['sku_x'] = map(str , sku_report['sku_x'].values)
    # sku_report['fdc'] = sku_report['fdc'].astype(str)
    # sku_report['date_s'] = pd.to_datetime(sku_report['date_s'])
    #
    # sku_list = list(sku_report['sku_x'].drop_duplicates())
    # for each in sku_list:
    #     # each = sku_list[0]
    #     sku_report_tmp = sku_report[sku_report['sku_x'] == each].loc[:,['date_s','inv','fdc']]
    #     plotinv(sku_report_tmp.sort(['date_s']))


