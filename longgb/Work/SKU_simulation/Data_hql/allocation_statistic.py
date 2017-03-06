#-*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pylab import mpl
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = ['SimHei']        # 使用中文
mpl.rcParams['axes.unicode_minus'] = False          # 解决负号问题

plt.rcParams['font.sans-serif'] = ['SimHei']        # 使用中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号问题


# ============================= 功能函数 =============================
def plothistper(plot_data, binsn, xlabeln, ylabeln, titlen, save_path='1', cum_True=True, size=(12,8), intshu=True, is_save=False):
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
    # plot_data=z_value_frame.z_value; binsn=[-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf]
    # xlabeln = u'z值'; ylabeln = u'频数'; titlen = u"Z值分布图"; size=(12,8); intshu=True
    ret = plt.hist(plot_data, bins=binsn, label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    counts, bins, patches = ret[0], ret[1], ret[2]
    if intshu:
        bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x,bins)
    bins_name = ["["+str(bins[i])+","+str(bins[i+1])+")" for i in range(len(bins)-1)]
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
    if is_save:
        plt.savefig(save_path)


# ============================= 使用函数 =============================
# ----------------------------- ss分布   -----------------------------
def readData():
    '''
    读取数据
    '''
    read_file = read_path + os.sep + file_name
    data_ss = pd.read_table(read_file, header=None)
    data_ss.columns = ['sku_id', 'stock_qtty', 'plan_allocation_qtty', 'forecast_sales_mean', 'plan_S', 'actual_S', 's', 'sale', 'sale_avg', 'date_s']
    data_ss['sku_id'] = data_ss['sku_id'].astype(str)
    data_ss['sale'] = data_ss['sale'].fillna(0)
    data_ss['sale_avg'] = data_ss['sale_avg'].fillna(0)
    return data_ss

def plotsSale(ax, title, x, y, data_ss):
    ax.plot(data_ss[x], data_ss[y], '.', alpha=0.2)
    ax.set_xlim(0,50)
    ax.set_ylim(-1,50)
    ax.set_title(title)

def plotlogsSale(ax, title, x, y, data_ss):
    ax.plot(data_ss[x], data_ss[y], '.', alpha=0.2)
    ax.set_xlim(0,50)
    # ax.set_ylim(-1,50)
    ax.set_title(title)

def plot_sSsale(data_ss):
    ss_Data = data_ss.loc[:,['sku_id','plan_S', 'actual_S','s','sale', 'sale_avg']]
    data_ss = data_ss.dropna()                              # 去掉所有空值

    # ======================= 产生负数原因 =======================
    ss_Data_neg = data_ss[data_ss['plan_S']<0]              # 为什么会有负数，原因: stock_qtty为负数，并且 inv + allocation < stock_qtty
    print 'Number of negative(plan_S) is : ', len(ss_Data_neg)
    print 'The sku_id is : '
    print ss_Data_neg['sku_id'].tolist()

    ss_Data_neg = data_ss[data_ss['actual_S']<0]
    print 'Number of negative(actual_S) is : ', len(ss_Data_neg)
    print 'The sku_id is : '
    print ss_Data_neg['sku_id'].tolist()

    ss_Data_neg = data_ss[data_ss['s']<0]                   # 为什么会有负数，原因， stock_qtty为负数，并且 inv < stock_qtty
    print 'Number of negative(s) is : ', len(ss_Data_neg)
    print 'The sku_id is : '
    print ss_Data_neg['sku_id'].tolist()

    # =============== plan 与 actual 的 S 之间的关系 ===============
    # ss_Data[ss_Data['plan_S'] != ss_Data['actual_S']]

    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(14, 10))

    plotsSale(ax1, r'plan_S * sale_avg', 'plan_S', 'sale_avg', data_ss)
    plotsSale(ax2, r'actual_S * sale_avg', 'actual_S', 'sale_avg', data_ss)
    plotsSale(ax3, r's * sale_avg', 's', 'sale_avg', data_ss)
    fig.savefig(save_path + os.sep + 'Plot_splan_Sactual_Ssale.png')

    fig, ax1 = plt.subplots(1, 1)
    plotsSale(ax1, r'plan_S * sale_avg', 'plan_S', 'sale_avg', data_ss)
    fig.savefig(save_path + os.sep + 'Plot_s_sale.png')

    fig, ax1 = plt.subplots(1, 1)
    plotsSale(ax1, r'actual_S * sale_avg', 'actual_S', 'sale_avg', data_ss)
    fig.savefig(save_path + os.sep + 'Plot_plan_S_sale.png')

    fig, ax1 = plt.subplots(1, 1)
    plotsSale(ax1, r's * sale_avg', 's', 'sale_avg', data_ss)
    fig.savefig(save_path + os.sep + 'Plot_actual_S_sale.png')

    binsn = [-np.inf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, np.inf]
    plothistper(data_ss['s'].dropna(), binsn, 's', 'sku_number', 's Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_s.png')
    binsn = [-np.inf, 0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, np.inf]
    plothistper(data_ss['plan_S'].dropna(), binsn, 'plan_S', 'sku_number', 'plan_S Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_plan_S.png', size=(16,8))
    plothistper(data_ss['actual_S'].dropna(), binsn, 'actual_S', 'sku_number', 'actual_S Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_actual_S.png', size=(16,8))

def cal_sSsale(data_ss):
    data_ss_upper = data_ss[data_ss['plan_S'] > 50]

def plot_log_sSsale(data_ss):
    ss_Data = data_ss.loc[:,['sku_id','plan_S', 'actual_S','s','sale', 'sale_avg']]
    data_ss = data_ss.dropna()                              # 去掉所有空值
    # data_ss_2 = data_ss.copy()
    data_ss['sale_log'] = np.log(data_ss['sale_avg'] + 0.001)

    # =============== plan 与 actual 的 S 之间的关系 ===============
    # ss_Data[ss_Data['plan_S'] != ss_Data['actual_S']]

    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(14, 10))

    plotlogsSale(ax1, r'plan_S * log(sale_avg)', 'plan_S', 'sale_log', data_ss)
    plotlogsSale(ax2, r'actual_S * log(sale_avg)', 'actual_S', 'sale_log', data_ss)
    plotlogsSale(ax3, r's * log(sale_avg)', 's', 'sale_log', data_ss)
    fig.savefig(save_path + os.sep + 'LOG_Plot_splan_Sactual_Ssale_log.png')

    fig, ax1 = plt.subplots(1, 1)
    plotlogsSale(ax1, r'plan_S * log(sale_avg)', 'plan_S', 'sale_log', data_ss)
    fig.savefig(save_path + os.sep + 'LOG_Plot_s_sale_log.png')

    fig, ax1 = plt.subplots(1, 1)
    plotlogsSale(ax1, r'actual_S * log(sale_avg)', 'actual_S', 'sale_log', data_ss)
    fig.savefig(save_path + os.sep + 'LOG_Plot_plan_S_sale_log.png')

    fig, ax1 = plt.subplots(1, 1)
    plotlogsSale(ax1, r's * log(sale_avg)', 's', 'sale_log', data_ss)
    fig.savefig(save_path + os.sep + 'LOG_Plot_actual_S_sale_log.png')

    # binsn = [-np.inf] + np.arange(0, 50, 4).tolist() + [50, np.inf]
    # plothistper(data_ss['s'].dropna(), binsn, 's', 'sku_number', 's Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_s.png')
    # plothistper(data_ss['plan_S'].dropna(), binsn, 'plan_S', 'sku_number', 'plan_S Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_plan_S.png')
    # plothistper(data_ss['actual_S'].dropna(), binsn, 'actual_S', 'sku_number', 'actual_S Distribution', is_save=True, save_path=save_path+os.sep+'Distribution_actual_S.png')

# ----------------------------- Ti分布   -----------------------------
def readDataTi():
    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')

    Tifile = 'ss_price_hql_create.out'
    read_file = read_path + os.sep + Tifile
    data_ss = pd.read_table(read_file, header=None)
    this_columns = ['rdc_id','fdc_id','sku_id','is_whitelist','plan_allocation_qtty','actual_allocation_qtty','forecast_daily_override_sales' ,'forecast_sales_mean','sale', 'sale_all', 'stock_qtty','date_s', 'price']
    data_ss.columns = this_columns
    data_ss['sku_id'] = data_ss['sku_id'].astype(str)
    # 画Ti的分布
    MAX_Ti = np.max(data_ss['sale_all']*data_ss['price'])
    # np.argmax(data_ss['sale_all']*data_ss['price'])
    # print data_ss['sale_all'][3660]*data_ss['price'][3660]
    Ti = map(lambda x: np.max([4, 2*np.sqrt(MAX_Ti * 1.0 / (x[0]*x[1]))]), data_ss.loc[:,['sale_all','price']].values)
    Ti = pd.DataFrame(Ti, columns=['Ti'])
    Ti = Ti.dropna()
    Ti = Ti[Ti['Ti'] != np.inf]
    binsn = [4, 5, 6, 7, 8, 9 ] + np.arange(10, 1000, 100).tolist() + [1000, np.inf]
    plothistper(Ti['Ti'],binsn,'number of SKU','Ti','Distribution_Ti',is_save=True, save_path=save_path+os.sep+'Distribution_Ti.png', size=(16,8))
    binsn = [4, 5, 6, 7, 8, 9 ] + np.arange(10, 100, 10).tolist() + np.arange(100, 1000, 200).tolist() + [1000, np.inf]
    plothistper(Ti['Ti'],binsn,'number of SKU','Ti','Distribution_Ti',is_save=True, save_path=save_path+os.sep+'Distribution_Ti_2.png', size=(19,8))
    print 'Max of Ti :', np.max(Ti)
    # 画Pi*Qi的分布
    piqi = map(lambda x: x[0]*x[1], data_ss.loc[:,['sale_all','price']].values)
    piqi = pd.DataFrame(piqi, columns=['piqi'])
    piqi = piqi.dropna()
    piqi = piqi[piqi['piqi'] != np.inf]
    print 'Median of Ti :', np.median(piqi['piqi'])
    print 'Mean of Ti :',np.mean(piqi['piqi'])
    binsn = np.arange(0, 1000, 100).tolist() + [1000, 4000, 7000, np.inf]
    plothistper(piqi['piqi'], binsn,'number of SKU','Ti','Distribution_PiQi',is_save=True, save_path=save_path+os.sep+'Distribution_piqi_big.png', size=(14,8))
    binsn = [0, 1] + np.arange(5, 100, 5).tolist()
    plothistper(piqi['piqi'], binsn,'number of SKU','Ti','Distribution_PiQi',is_save=True, save_path=save_path+os.sep+'Distribution_piqi_litte.png', size=(16,8))

    # ret = plt.hist(Ti, bins=binsn, label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    # counts, bins, patches = ret[0], ret[1], ret[2]
    # data_ss['sale'] = data_ss['sale'].fillna(0)
    # data_ss['sale_avg'] = data_ss['sale_avg'].fillna(0)
    pass


# ======================= 参数设置 =======================
read_path = r'D:\Lgb\data_sz'
save_path = r'D:\Lgb\WorkFiles\SKU_Allocations'
file_name = 'sS_hql.out'


if __name__ == '__main__':
    # data_ss = readData()
    # plot_sSsale(data_ss)
    # plot_log_sSsale(data_ss)
    readDataTi()
    pass


