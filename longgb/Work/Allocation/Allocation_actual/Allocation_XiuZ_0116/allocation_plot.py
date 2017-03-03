#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import warnings
warnings.filterwarnings('ignore')


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


def plot_kpi_ito(Kpi_report_tmp, save_name, size=(18, 12), step=1, lins=20, alpha=0):
    # Kpi_report_plot = Kpi_report.loc[:,['fdc', 'date_s','satisfy_rate_actual','fill_rate_sim', 'ito_actual','ito_rate_sim']]
    # Kpi_report_tmp['delta'] = Kpi_report_tmp['ito_sim'] - Kpi_report_tmp['ito_actual']
    Kpi_report_plot_fill = Kpi_report_tmp['ito_sim'] - Kpi_report_tmp['ito_actual']
    # np.max(Kpi_report_plot_fill) = 965.5
    # np.min(Kpi_report_plot_fill) = -501.66666666666669
    # Kpi_report_tmp.loc[np.argmax(Kpi_report_plot_fill),:]  =  sku : 1577266
    # Kpi_report_tmp.loc[np.argmin(Kpi_report_plot_fill),:]  =  sku : 3067997
    # len(Kpi_report_plot_fill[Kpi_report_plot_fill > upper])       1142
    # len(Kpi_report_plot_fill[Kpi_report_plot_fill < lower])       1146
    # len(Kpi_report_plot_fill)
    lower = np.nanpercentile(Kpi_report_plot_fill, alpha)
    upper = np.nanpercentile(Kpi_report_plot_fill, 100 - alpha)
    Kpi_report_plot_fill2 = Kpi_report_plot_fill[(Kpi_report_plot_fill>lower)&(Kpi_report_plot_fill<upper)]
    Kpi_report_tmp_return = Kpi_report_tmp[(Kpi_report_plot_fill<lower)|(Kpi_report_plot_fill>upper)]
    max_num = int(np.ceil(Kpi_report_plot_fill2.max()))*1.0
    min_num = int(np.floor(Kpi_report_plot_fill2.min()))*1.0
    binsn1 = np.arange(min_num,0,step)
    binsn2 = np.arange(0,max_num,step)
    binsn = np.concatenate([binsn1, binsn2], axis=0)
    # binsn = np.arange(min_num,max_num,step)
    binsn = [-np.inf] + list(binsn) + [np.inf]
    # binsn = np.linspace(min_num,max_num,lins)
    xlabeln = u'差值'
    ylabeln = u'SKU数'
    titlen = u'{0} （仿真周转 - 实际周转）差值图 '.format(save_name)
    intshu = True if isinstance(step, int) else False
    plothistper(Kpi_report_plot_fill, binsn, xlabeln, ylabeln, titlen, save_path + os.sep + 'Kpi_report_ito' + save_name + '.png', size=size, intshu=intshu)
    return Kpi_report_tmp_return


# =======================================================================
# =                                 路径设置                            =
# =======================================================================
# 路径
report_path = r'D:\Lgb\data_local'
save_path = r'D:\Lgb\data_local'


if __name__ == '__main__':
    # ===============================================================================
    # =                                 （1）Kpi 数据表                             =
    # ===============================================================================
    # 1、读取数据
    # Kpi_report = pd.read_table(report_path + os.sep + 'sku_ito.csv', header=None)
    Kpi_report = pd.read_table(report_path + os.sep + 'sku_ito_new_xiugai.result', header=None)
    Kpi_report.columns = ['fdc', 'sku_id','inv_actual','sales_actual','ito_actual','inv_sim','sales_sim','ito_sim']
    Kpi_report['fdc'] = Kpi_report['fdc'].astype(str)
    type_sure = Kpi_report.columns[2:]
    Kpi_report = Kpi_report.dropna()
    for each in type_sure:
        Kpi_report[each] = Kpi_report[each].astype(float)

    # Kpi_report_tmp.to_csv(save_path + os.sep + 'aa.csv', index=False)
    Kpi_report_return = []
    for fdc in ['628', '630', '658']:
        # Kpi_report_tmp = Kpi_report[Kpi_report['fdc'] == '628']
        Kpi_report_tmp = Kpi_report[Kpi_report['fdc'] == fdc]
        # plot_kpi_fill(Kpi_report_tmp, fdc)
        Kpi_report_return.append(plot_kpi_ito(Kpi_report_tmp, fdc, size=(18, 12), step=4, alpha=4))
        # plot_kpi_ito(Kpi_report_tmp, fdc, size=(12, 8), step=0.5)
    # plot_kpi_fill(Kpi_report, 'Total', size=(18, 12), step=0.02)
    plot_kpi_ito(Kpi_report, 'Total', size=(18, 12), step=4, alpha=4)

    sku628 = Kpi_report_return[0]
    sku628['delta'] = sku628['ito_sim'] - sku628['ito_actual']
    sku628.to_csv(save_path + os.sep + '628_per4.csv', index=False)

    # 抽数据问题
    # 这个地方的问题?
    data_tmp_upper = sku628[sku628['delta'] > 0]
    data_tmp_lower = sku628[sku628['delta'] < 0]
    # data_tmp[data_tmp['sku_id'] == 2967897]
    tmp_upper = pd.DataFrame(list(np.random.choice(data_tmp_upper['sku_id'],20)), columns=['sku'])
    tmp_lower = pd.DataFrame(list(np.random.choice(data_tmp_lower['sku_id'],20)), columns=['sku'])
    tmp_upper.to_csv(save_path + os.sep + '628_per4_upper.csv', index=False)
    tmp_lower.to_csv(save_path + os.sep + '628_per4_lower.csv', index=False)

    upper = map(str, tmp_upper['sku'].values)
    lower = map(str, tmp_lower['sku'].values)
    print upper
    print lower

    # ['1851648', '3039357', '1038233', '557624', '1593768', '2878882', '563023', '1044553', '659701', '1072640', '3515938', '1023596', '1833441', '3149290', '1633245', '2591599', '521597', '935350', '1577112', '2327491']
    # ['829332', '884814', '1432063', '947385', '1189528', '3149957', '851499', '1094467', '1516565', '2341397', '1577165', '1636826', '1329098', '3544036', '234284', '1340212', '1793724', '366776', '896109', '1274678']


Kpi_report_tmp.columns
Kpi_report_tmp_upper = Kpi_report_tmp[Kpi_report_tmp['delta'] >= 0]
np.sum(Kpi_report_tmp_upper['inv_actual'])      # 2439522.0
np.sum(Kpi_report_tmp_upper['sales_actual'])    # 191392.0
np.sum(Kpi_report_tmp_upper['inv_sim'])         # 2789272.0
np.sum(Kpi_report_tmp_upper['sales_sim'])       # 163114.0
Kpi_report_tmp_lower = Kpi_report_tmp[Kpi_report_tmp['delta'] < 0]
np.sum(Kpi_report_tmp_lower['inv_actual'])      # 3039382.0
np.sum(Kpi_report_tmp_lower['sales_actual'])    # 289607.0
np.sum(Kpi_report_tmp_lower['inv_sim'])         # 2301499.0
np.sum(Kpi_report_tmp_lower['sales_sim'])       # 279284.0


(2439522+3039382)*1.0/(191392+289607)       # 11.390676487892906
(2789272+2301499)*1.0/(163114+279284)       # 11.507219743308061


