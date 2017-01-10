#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import seaborn as sns


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


def plot_kpi_fill(Kpi_report, save_name):
    # Kpi_report_plot = Kpi_report.loc[:,['fdc', 'date_s','satisfy_rate_actual','fill_rate_sim', 'ito_actual','ito_rate_sim']]
    Kpi_report_plot_fill = Kpi_report['fill_rate_sim'] - Kpi_report['satisfy_rate_actual']
    max_num = int(np.ceil(Kpi_report_plot_fill.max() * 100))*1.0 / 100
    min_num = int(np.floor(Kpi_report_plot_fill.min() * 100))*1.0 / 100
    binsn = np.arange(min_num,max_num,0.02)
    xlabeln = u'差值'
    ylabeln = u'天数'
    titlen = u'{0} （仿真满足率 - 实际满足率）差值图 '.format(save_name)
    plothistper(Kpi_report_plot_fill, binsn, xlabeln, ylabeln, titlen, save_path + os.sep + 'Kpi_report_fill_rate' + save_name + '.png', intshu=False)


def plot_kpi_ito(Kpi_report, save_name):
    # Kpi_report_plot = Kpi_report.loc[:,['fdc', 'date_s','satisfy_rate_actual','fill_rate_sim', 'ito_actual','ito_rate_sim']]
    Kpi_report_plot_fill = Kpi_report['ito_rate_sim'] - Kpi_report['ito_actual']
    max_num = int(np.ceil(Kpi_report_plot_fill.max()))*1.0
    min_num = int(np.floor(Kpi_report_plot_fill.min()))*1.0
    binsn = np.arange(min_num,max_num,0.5)
    xlabeln = u'差值'
    ylabeln = u'天数'
    titlen = u'{0} （仿真周转 - 实际周转）差值图 '.format(save_name)
    plothistper(Kpi_report_plot_fill, binsn, xlabeln, ylabeln, titlen, save_path + os.sep + 'Kpi_report_ito' + save_name + '.png', size=(18, 12), intshu=False)



# =======================================================================
# =                                 路径设置                            =
# =======================================================================
# 总路径
path_sim = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3'           # 读取仿真数据的总路径
path_actual = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3'        # 实际数据的总路径，[ 不用改 ]
# 实际数据的路径 [ 不用改 ]
fill_actual_path = path_actual + os.sep + 'full_actual.xls'
ito_actual_path = path_actual + os.sep + 'ito_actual.xls'
ito_sku_nodate_actual_path = path_actual + os.sep + 'sku_ito'
# 仿真数据路径
table_fill_path = path_sim + os.sep + 'table_fill.csv'
table_ito_path = path_sim + os.sep + 'table_ito.csv'
table_ito_sku_nodate_path = path_sim + os.sep + 'table_ito_sku_nodate.csv'
# 储存报表的路径
save_path = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3'          # 存储的路径


if __name__ == '__main__':
    # ===============================================================================
    # =                                 （1）Kpi 数据表                             =
    # ===============================================================================
    # 1、读取数据
    # 实际数据
    fill_actual = pd.read_excel(fill_actual_path)
    fill_actual.columns = ['fdc', 'date_s', 'rdc_order_num_actual', 'total_order_num_actual', 'satisfy_rate_actual']
    fill_actual['fdc_order_num_actual'] = fill_actual['total_order_num_actual'] - fill_actual['rdc_order_num_actual']
    fill_actual_need = fill_actual.loc[:,['fdc', 'date_s', 'fdc_order_num_actual', 'total_order_num_actual', 'satisfy_rate_actual']]
    ito_actual = pd.read_excel(ito_actual_path, sheetname='Sheet1')
    # print ito_actual
    ito_actual.columns = ['fdc', 'date_s', 'inv_actual', 'sale_qtty_actual', 'ito_actual']
    # 仿真数据
    table_fill = pd.read_csv(table_fill_path)
    table_fill.columns = ['fdc', 'date_s', 'fdc_order_num_sim', 'total_order_num_sim', 'fill_rate_sim']
    table_ito = pd.read_csv(table_ito_path)
    table_ito.columns = ['date_s', 'fdc', 'inv_sim', 'sale_qtty_sim', 'ito_rate_sim']

    # 2、合并表 Kpi_report
    Kpi_report = fill_actual_need.merge(table_fill, on=['fdc', 'date_s'])
    Kpi_report['loss_order'] = Kpi_report['total_order_num_actual'] - Kpi_report['total_order_num_sim']
    Kpi_report['loss_order_rate'] = Kpi_report['loss_order'] * 1.0 / Kpi_report['total_order_num_actual']
    Kpi_report = Kpi_report.merge(ito_actual, on=['fdc', 'date_s'])
    Kpi_report = Kpi_report.merge(table_ito, on=['fdc', 'date_s'])

    # 3、保存报表
    Kpi_report.to_csv(save_path + os.sep + 'Kpi_report.csv', index=False)

    # for fdc in [628, 630, 658]:
    #     Kpi_report_tmp = Kpi_report[Kpi_report['fdc'] == fdc]
    #     plot_kpi_fill(Kpi_report_tmp, str(fdc))
    #     plot_kpi_ito(Kpi_report_tmp, str(fdc))
    # plot_kpi_fill(Kpi_report, '')
    # plot_kpi_ito(Kpi_report, '')


    # ===============================================================================
    # =                                 （1）Sku 粒度表                              =
    # ===============================================================================
    # 1、读取数据
    # 仿真数据
    ito_sku_nodate_actual = pd.read_table(ito_sku_nodate_actual_path, header=None)
    ito_sku_nodate_actual.columns = ['fdc', 'sku', 'inv_actual', 'sale_qtty_actual', 'ito_rate_actual']
    ito_sku_nodate_actual['fdc'] = ito_sku_nodate_actual['fdc'].astype(str)
    ito_sku_nodate_actual['sku'] = ito_sku_nodate_actual['sku'].astype(str)
    # 实际数据
    table_ito_sku_nodate = pd.read_csv(table_ito_sku_nodate_path)
    table_ito_sku_nodate.columns = ['sku', 'fdc', 'inv_sim', 'sale_qtty_sim', 'ito_rate_sim']
    table_ito_sku_nodate['fdc'] = table_ito_sku_nodate['fdc'].astype(str)
    table_ito_sku_nodate['sku'] = table_ito_sku_nodate['sku'].astype(str)

    # 2、合并表 Sku_report
    Sku_report = ito_sku_nodate_actual.merge(table_ito_sku_nodate, on=['fdc', 'sku'])
    Sku_report2 = ito_sku_nodate_actual.merge(table_ito_sku_nodate, on=['fdc', 'sku'], how='left')

    # 3、保存报表
    Sku_report.to_csv(save_path + os.sep + 'Sku_report.csv', index=False)
