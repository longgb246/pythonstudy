# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/16
  Usage   : 
"""

import os
from string import Template
from dateutil.parser import parse
import datetime
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 使用tk画图

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dateutil.rrule import rrule, DAILY

dt = '2018-09-16'
path = '/Users/longguangbin/Work/scripts/anta_offline/detail'
sale_file = 'sale_active.tsv'
pre_file = 'pred_{0}.tsv'.format(dt)

# sku_code = '11721360-7/8.5'
# 问题 sku：15821204-3/2XL - KL00 | 15821185-3/M_K50N | 19827252-2_K50P
sku_code = '__all__' # __all__ | 19827252-2
# store_id = 'L611'
store_id = '__all__'  # '__all__'
models = ['combine']  # reg_single | hw | wma | combine
pre_len = 7
before_day = 90
show_qty = True


def get_date_range1(date_start, pre_len):
    date_start_dt = parse(date_start)
    return [(date_start_dt + datetime.timedelta(x + 1)).strftime('%Y-%m-%d') for x in range(pre_len)]


def get_all_date(date_begin, date_end):
    date_begin_dt = parse(date_begin)
    date_end_dt = parse(date_end)
    date_len = (date_end_dt - date_begin_dt).days + 1
    return [(date_begin_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range(date_len)]


def get_data(dt, path, sale_file, pre_file, sku_code, store_id, pre_len, models):
    pre_data = pd.read_table(path + os.sep + pre_file)
    real_data = pd.read_table(path + os.sep + sale_file)

    group_store = store_id.lower() == '__all__'
    group_sku = sku_code.lower() == '__all__'

    date_range = get_date_range1(dt, pre_len)

    if group_store and group_sku:
        tmp_real = real_data.groupby(['dt']).agg({'sale': 'sum'}).reset_index()
    else:
        tmp_real = real_data[(real_data['sku_code'].apply(lambda x: sku_code in x)) & (
            real_data['store_id'].apply(lambda x: True if group_store else store_id == x))]
        # tmp_real = tmp_real.groupby(['sku_code', 'dt']).agg({'qty': 'sum', 'sale': 'sum'}).reset_index()
        tmp_real = tmp_real.groupby(['sku_code', 'dt']).agg({'sale': 'sum'}).reset_index()
        # tmp_real = tmp_real.groupby(['dt']).agg({'qty': 'sum', 'sale': 'sum'}).reset_index()
        tmp_real = tmp_real.groupby(['dt']).agg({'sale': 'sum'}).reset_index()

    if len(tmp_real) < 1:
        raise Exception(''' tmp_real is empty : {0} '''.format(tmp_real))

    dt_min, dt_max = min([np.min(tmp_real['dt'])] + date_range), max([np.max(tmp_real['dt'])] + date_range)
    all_date_range = get_all_date(dt_min, dt_max)
    date_range_df = pd.DataFrame(all_date_range, columns=['dt'])

    pre_list = []
    for model in models:
        if group_store and group_sku:
            tmp_pre = pre_data[pre_data['sale_type'].apply(lambda x: model == x)]
            tmp_value = np.sum(map(lambda x: eval(x), tmp_pre['sale_list'].values), axis=0)
        else:
            tmp_pre = pre_data[(pre_data['sku_code'].apply(lambda x: sku_code in x)) & (
                pre_data['store_id'].apply(lambda x: True if group_store else store_id == x)) & (
                                   pre_data['sale_type'].apply(lambda x: model == x))]
            tmp_pre = tmp_pre.groupby(['sku_code']).agg(
                {'sale_list': lambda y: str(np.sum(map(lambda x: eval(x), y.values), axis=0).tolist())}).reset_index()
            tmp_value = tmp_pre['sale_list'].values
        if len(tmp_value) == 0:
            tmp_value = [0] * pre_len
        else:
            tmp_value = np.sum(map(lambda x: eval(x), tmp_value), axis=0).tolist()
        pre_values = tmp_value[:pre_len]
        pre_df = pd.DataFrame(pre_values, columns=['sale'])
        pre_df['dt'] = date_range
        pre_list.append(pre_df)

    # real_sale_df = date_range_df.merge(tmp_real.loc[:, ['sale', 'qty', 'dt']], on=['dt'], how='left').fillna(0)
    real_sale_df = date_range_df.merge(tmp_real.loc[:, ['sale', 'dt']], on=['dt'], how='left').fillna(0)

    return real_sale_df, pre_list, dt_min, dt_max


def get_weekend(start_date, end_date):
    this_day = parse(start_date).weekday()
    start_date = parse(start_date)
    end_date = parse(end_date)

    sat_delta = 6 if this_day > 5 else 5 - this_day
    sun_delta = 6 - this_day

    sat_day = start_date + datetime.timedelta(sat_delta)
    sun_day = start_date + datetime.timedelta(sun_delta)

    day_list = []
    while (sat_day < end_date) or (sun_day < end_date):
        if sat_day < end_date:
            day_list.append(sat_day)
            sat_day = sat_day + datetime.timedelta(7)
        if sun_day < end_date:
            day_list.append(sun_day)
            sun_day = sun_day + datetime.timedelta(7)
    week_list = sorted(map(lambda x: x.strftime('%Y-%m-%d'), day_list))
    return week_list


def get_date_list(start_date, day_len):
    date_list = map(lambda x: x.strftime('%Y-%m-%d'),
                    list(rrule(freq=DAILY, count=day_len, dtstart=parse(start_date) + datetime.timedelta(1))))
    return date_list


def get_date_range(start_date, end_date):
    start_date_dt = parse(start_date)
    end_date_dt = parse(end_date)
    date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime("%Y-%m-%d"),
                     range((end_date_dt - start_date_dt).days + 1))
    return date_range


def get_week_df(start_date, end_date):
    week_list = get_weekend(start_date, end_date)
    date_list = get_date_range(start_date, end_date)
    date_pd = pd.DataFrame(date_list, columns=['dt'])
    week_pd = pd.DataFrame(week_list, columns=['dt'])
    week_pd['week'] = '1'
    week_df = date_pd.merge(week_pd, on=['dt'], how='left').fillna('0')
    return week_df


def plot_func(start_date=None, before_day=30, real_sale=None, data_list=None, name_list=None, qty=True, dt_min=None,
              dt_max=None, sku_code=None, store_id=None):
    before_date = (parse(start_date) - datetime.timedelta(before_day)).strftime('%Y-%m-%d')
    dt_min_min = before_date
    week_df = get_week_df(dt_min_min, dt_max)
    tmp_sale = real_sale[real_sale['dt'] > dt_min_min]
    all_date_range = get_all_date(dt_min_min, dt_max)
    date_range_df = pd.DataFrame(all_date_range, columns=['dt'])
    # real_sale = date_range_df.merge(tmp_sale.loc[:, ['sale', 'qty', 'dt']], on=['dt'], how='left').fillna(0)
    real_sale = date_range_df.merge(tmp_sale.loc[:, ['sale', 'dt']], on=['dt'], how='left').fillna(0)
    # tmp_sale = sale_sum[sale_sum['dt'] > before_date]
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    if qty:
        ax.plot(real_sale['dt'], real_sale['qty'], label='qty', alpha=0.9)
    ax.plot(real_sale['dt'], real_sale['sale'], label='real')
    name_list = ['dev'] * len(data_list) if name_list is None else name_list
    have_pre = data_list is not None and len(data_list) > 0
    if have_pre:
        for i, each in enumerate(data_list):
            ax.plot(each['dt'], each['sale'], label=name_list[i])
    x_tick_labels = list(ax.get_xticklabels())
    tick_num = 10  # 刻度数目
    tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    # print x_labels, tick_spacing
    y_max = max([np.max(real_sale['sale'])] + map(lambda x: np.max(x['sale']), data_list)) if have_pre else np.max(
        real_sale['sale'])
    y_min = min([np.min(real_sale['sale'])] + map(lambda x: np.min(x['sale']), data_list)) if have_pre else np.min(
        real_sale['sale'])
    y_gap = y_max - y_min
    width = 1
    ax.bar(week_df['dt'], week_df['week'].apply(lambda x: y_max + y_gap * 0.2 if x == '1' else 0), width, color="red",
           align='center', alpha=0.15)
    # ax.yaxis.grid(False)
    ax.set_ylim(y_min - y_gap * 0.03, y_max + y_gap * 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.55))
    ax.set_title('Sku_code : {2} Store_id : {3} \nPredict date : {0} . History Windows : {1} days.'.
                 format(start_date, before_day, sku_code, store_id),
                 fontsize=15)
    plt.show()


real_sale_df, pre_list, dt_min, dt_max = get_data(dt, path, sale_file, pre_file, sku_code, store_id, pre_len, models)

plot_func(start_date=dt, before_day=before_day, real_sale=real_sale_df, data_list=pre_list, name_list=models,
          qty=show_qty, dt_min=dt_min, dt_max=dt_max, sku_code=sku_code, store_id=store_id)


# 衡量数据 稀疏度+连续度+不平衡度
# 稀疏度：有销量天数 / base_day
# 连续度：max 有销量的连续长度
# 不平衡度：sale_sum

# base_day：45

# 均销量高：((sale_sum / 有销量天数) > 4 )

# 数据稀疏：((有销量天数 / base_day) < 0.2 )

# sku 角度：
# sku + store ：总销量排序


def plotHistPer(plot_data, binsn=[], xlabeln='x', ylabeln='y', titlen='', save_path='', cum_True=True, size=(12, 8),
                detail=0, is_drop_zero=False, is_show=True, p_detail=2, sp_0=False):
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
    plot_data = list(plot_data)
    plt.style.use('seaborn-darkgrid')
    if binsn == []:
        ret = plt.hist(plot_data, label='Z', color='#0070C0', histtype='bar', rwidth=0.6)
    else:
        ret = plt.hist(plot_data, bins=binsn, label='Z', color='#0070C0', histtype='bar', rwidth=0.6)
    plt.close()
    counts, bins, patches = ret[0], ret[1], ret[2]
    detail_method = Template("[{0:.${detail}f},{1:.${detail}f})")
    bins_name = [detail_method.substitute(detail=detail).format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    if sp_0:
        bins_name[0] = '(' + bins_name[0][1:]
        bins_name = ['[0]'] + bins_name
        zero_cnt = sum([1 if n_i == 0 else 0 for n_i in plot_data])
        counts = list(counts)
        counts[0] = counts[0] - zero_cnt
        counts = [zero_cnt] + counts
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
    ax1.bar(ind + width2, counts, width, color="#0070C0", tick_label=bins_name, align='center', alpha=0.8)
    counts_per = counts / np.sum(counts)
    counts_per_cum = np.cumsum(counts_per)
    i = 0
    ymin, ymax = plt.ylim()
    ax1.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
    # ax1.set_xlim(-1, len(bins_name)+1)
    percent_detail = Template("{0:.${p_detail}f}%").substitute(p_detail=p_detail)
    for x, y in zip(ind, counts):
        ax1.text(x + width2, y + 0.05, percent_detail.format(counts_per[i] * 100), ha='center', va='bottom')
        i += 1
    plt.title(titlen)
    if cum_True:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative probability distribution')
        ax2.plot(ind + width2, counts_per_cum, '--', color="red")
        ax2.yaxis.grid(False)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(-0.5, len(bins_name) - 0.5)
    if save_path != '':
        plt.savefig(save_path)
    if is_show:
        plt.show()
    return [fig1, ax1, ax2]


def plot_real_data(data, pre_list=None, name_list=None, start_date=None, before_day=None, sku_code=None, store_id=None):
    before_date = (parse(start_date) - datetime.timedelta(before_day)).strftime('%Y-%m-%d')
    dt_min_min = before_date
    week_df = get_week_df(dt_min_min, dt_max)
    all_date_range = get_all_date(dt_min_min, dt_max)
    date_range_df = pd.DataFrame(all_date_range, columns=['dt'])
    real_sale = date_range_df.merge(data.loc[:, ['sale', 'dt']], on=['dt'], how='left').fillna(0)
    # tmp_sale = sale_sum[sale_sum['dt'] > before_date]
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(real_sale['dt'], real_sale['sale'], label='real')
    have_pre = pre_list is not None and len(pre_list) > 0
    if have_pre:
        for i, each in enumerate(pre_list):
            ax.plot(each['dt'], each['sale'], label=name_list[i])
    x_tick_labels = list(ax.get_xticklabels())
    tick_num = 10  # 刻度数目
    tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    # print x_labels, tick_spacing
    y_max = np.max(real_sale['sale'])
    y_min = np.min(real_sale['sale'])
    y_gap = y_max - y_min
    width = 1
    ax.bar(week_df['dt'], week_df['week'].apply(lambda x: y_max + y_gap * 0.2 if x == '1' else 0), width, color="red",
           align='center', alpha=0.15)
    # ax.yaxis.grid(False)
    ax.set_ylim(y_min - y_gap * 0.03, y_max + y_gap * 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.55))
    ax.set_title('Sku_code : {2} Store_id : {3} \nPredict date : {0} . History Windows : {1} days.'.
                 format(start_date, before_day, sku_code, store_id),
                 fontsize=15)
    plt.show()


pre_data = pd.read_table(path + os.sep + 'pred_2018-08-04.tsv')
real_data = pd.read_table(path + os.sep + 'sale_active.tsv')
cate_data = pd.read_table(path + os.sep + 'cate_active.tsv')
loc_data = pd.read_table(path + os.sep + 'loc_active.tsv')

this_date = '2018-08-04'
delay_days = 60
start_date = (parse(this_date) - datetime.timedelta(delay_days)).strftime('%Y-%m-%d')
tmp_data = real_data.loc[(real_data['dt'] > start_date) & (real_data['dt'] < this_date),
                         ['sku_code', 'store_id', 'sale']]
sale_sum = tmp_data.groupby(['sku_code', 'store_id']).agg({'sale': 'sum'}).reset_index(). \
    sort_values('sale', ascending=False)
keep_sum = sale_sum[sale_sum['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))]

# pd.set_option('display.max_columns', 40)
# pd.set_option('display.width', 180)
keep_sum.merge(cate_data.loc[:, ['sku_code', 'sku_name', 'spu_name']], on=['sku_code']). \
    sort_values('sale', ascending=False)

## 1、real_data 数据需要剔除塑料袋 XSTD | FSTD
# mm = real_data.groupby(['sku_code']).agg({'sale': 'sum'}).reset_index(). \
#     sort_values('sale', ascending=False)
# mm.index = range(len(mm))
cond_1 = real_data['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))
## 2、只看 180 天至分析日以后的数据，分析日：this_date = '2018-08-20'
this_date = '2018-08-04'
delay_days = 180
start_date = (parse(this_date) - datetime.timedelta(delay_days)).strftime('%Y-%m-%d')
cond_2 = (real_data['dt'] > start_date) & (real_data['dt'] < this_date)
real_keep = real_data[cond_1 & cond_2]
real_keep_no = real_data[cond_2]
real_keep_no['sku_code'].drop_duplicates()
real_keep_no['store_id'].drop_duplicates()
real_keep_no['cate1'].drop_duplicates()
len(real_keep_no['cate2'].drop_duplicates())
real_keep_no.loc[:, ['sku_code', 'store_id']].drop_duplicates()
# cate1:5, 38

## 查看 sales 分布情况
sale_sum = real_keep.groupby(['sku_code', 'store_id']).agg({'sale': 'sum'}).reset_index(). \
    sort_values('sale', ascending=False)
# key : 199376 (23865 sku) '2018-08-01'
# key : 188897 (22596 sku) '2018-08-20' - 180
plotHistPer(sale_sum['sale'], binsn=[1, 2, 3, 5, 7, 9, 11, 15, 30, 50, 180, np.inf], xlabeln='sales',
            ylabeln='count number', titlen='( {0} ~ {1} ) Sales Distribution'.format(start_date, this_date),
            save_path='', cum_True=True, size=(12, 8), detail=0, is_drop_zero=False, is_show=True)

## 查看 sales_days 分布情况
sales_days = real_keep.groupby(['sku_code', 'store_id']).agg({'dt': 'count'}).reset_index(). \
    sort_values('dt', ascending=False)
plotHistPer(sales_days['dt'], binsn=[1, 2, 3, 5, 7, 9, 11, 15, 30, 50, 90, 180, np.inf], xlabeln='dt_cnt',
            ylabeln='count number', titlen='( {0} ~ {1} ) Sales Days Distribution'.format(start_date, this_date),
            save_path='', cum_True=True, size=(12, 8), detail=0, is_drop_zero=False, is_show=True, p_detail=3)

## 查看个例 sku 的情况
dt = '2018-08-04'
path = '/Users/longguangbin/Work/scripts/anta_offline/detail'
sale_file = 'sale_active.tsv'
pre_file = 'pred_{0}.tsv'.format(dt)
# sale_sum[sale_sum['sale'] == 50]  19827252-2_K50P
sku_code, store_id = '19827252-2', 'K50P'
# sales低：15821783-3/L + K50M,  11827711-1/8 + L638
# sales中等：19817361-2 + K515, 15821204-3/2XL + KL00
# sales高：
# 19746302-1_KLA4
# store_id = 'L64C'  # '__all__'
models = ['reg_single', 'hw', 'wma', 'combine']  # reg_single | hw | wma | combine
pre_len = 7
before_day = delay_days
show_qty = False
sample_1 = real_keep[(real_keep['sku_code'] == sku_code) & (real_keep['store_id'] == store_id)]
real_sale_df, pre_list, dt_min, dt_max = get_data(dt, path, sale_file, pre_file, sku_code, store_id, pre_len, models)
dt_max = (parse(dt) + datetime.timedelta(pre_len)).strftime('%Y-%m-%d')
pre_list = None
plot_func(start_date=dt, before_day=before_day, real_sale=real_sale_df, data_list=pre_list, name_list=models,
          qty=show_qty, dt_min=dt_min, dt_max=dt_max, sku_code=sku_code, store_id=store_id)


def plot_for(sku_code=None, store_id=None):
    dt = '2018-08-04'
    path = '/Users/longguangbin/Work/scripts/anta_offline/detail'
    sale_file = 'sale_active.tsv'
    pre_file = 'pred_{0}.tsv'.format(dt)
    models = ['reg_single', 'hw', 'wma', 'combine']  # reg_single | hw | wma | combine
    pre_len = 7
    before_day = delay_days
    show_qty = False
    sample_1 = real_keep[(real_keep['sku_code'] == sku_code) & (real_keep['store_id'] == store_id)]
    real_sale_df, pre_list, dt_min, dt_max = get_data(dt, path, sale_file, pre_file, sku_code, store_id, pre_len,
                                                      models)
    dt_max = (parse(dt) + datetime.timedelta(pre_len)).strftime('%Y-%m-%d')
    # pre_list = None
    plot_func(start_date=dt, before_day=before_day, real_sale=real_sale_df, data_list=pre_list, name_list=models,
              qty=show_qty, dt_min=dt_min, dt_max=dt_max, sku_code=sku_code, store_id=store_id)


m_list = [['731300056', 'KL2A'], ['19827308-3', 'KL30'], ['19827303-3', 'KL5B'], ['19827308-3', 'KL5H'],
          ['19827306-6', 'KL3S'], ['15823504-2/XL', 'K50F'], ['15823504-2/L', 'K50F'], ['19827303-3', 'KL5H'],
          ['15821742-3/XL', 'K50W'], ['19827306-1', 'KL5B'], ['19827308-1', 'KL5B'], ['19825303-2', 'KL0C'],
          ['19827303-2', 'KL5B'], ['19827306-6', 'KL5A'], ['15821742-3/L', 'K50W'], ['731100105', 'KL54'],
          ['16827160-2/L', 'K507'], ['19827301-3', 'KL5B'], ['19827308-2', 'KL0H'], ['19827308-3', 'KL3A'],
          ['19827308-2', 'KL1G'], ['19817311-3', 'K501'], ['19825304-5', 'KL0C'], ['19825304-4', 'KL0C'],
          ['19825303-3', 'KL3D'], ['19817311-2', 'L64G'], ['19827301-3', 'KL5H'], ['15821742-3/XL', 'K55P']]

for v in m_list:
    sku_code = v[0]
    store_id = v[1]
    plot_for(sku_code=sku_code, store_id=store_id)

## 查看 gap 的分布情况
gap_df = pd.read_csv(path + os.sep + 'gap_sp2.csv', header=None)
gap_df.columns = ['sku_code', 'sale', 'gap']

gap_keep = gap_df[gap_df['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))]

plotHistPer(gap_keep['gap'], binsn=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, np.inf], xlabeln='gap',
            ylabeln='count number', titlen='( {0} ~ {1} ) Sales Gap Distribution'.format(start_date, this_date),
            save_path='', cum_True=True, size=(12, 8), detail=0, is_drop_zero=False, is_show=True, p_detail=3, sp_0=True)



gap_keep[gap_keep['gap']>5].sort_values('gap', ascending=False)


real_data_tmp = real_data[(real_data['dt'] >= '2018-02-05') & (real_data['dt'] <= '2018-08-11')]
real_data_tmp = real_data_tmp.loc[:, ['sku_code', 'store_id']].drop_duplicates()
real_data_tmp = real_data_tmp[real_data_tmp['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))]
real_data_tmp.index = range(len(real_data_tmp))

be_data = real_data[(real_data['dt'] >= '2018-07-16') & (real_data['dt'] <= '2018-08-04')].groupby(['sku_code', 'store_id']).agg({'sale': 'sum'}).reset_index()
af_data = real_data[(real_data['dt'] >= '2018-08-05') & (real_data['dt'] <= '2018-08-11')].groupby(['sku_code', 'store_id']).agg({'sale': 'sum'}).reset_index()
be_data = be_data[be_data['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))]
af_data = af_data[af_data['sku_code'].apply(lambda x: not any(map(lambda y: y in x, ['XSTD', 'FSTD'])))]
# join_data = be_data.merge(af_data, on=['sku_code', 'store_id'], how='left').fillna(0)
join_data = real_data_tmp.merge(be_data, on=['sku_code', 'store_id'], how='left').\
    merge(af_data, on=['sku_code', 'store_id'], how='left').fillna(0)


# 11827711-1/8   L638

def map_score(x):
    if x == 0:
        return '0'
    elif x <= 1 :
        return '1'
    elif x <= 2 :
        return '2'
    elif x <= 3 :
        return '3'
    elif x <= 4 :
        return '4'
    elif x <= 5 :
        return '5'
    else:
        return '5>'

join_data['be_flag'] = join_data['sale_x'].apply(lambda x: map_score(x))
join_data['af_flag'] = join_data['sale_y'].apply(lambda x: map_score(x))

import seaborn as sns
from collections import Counter

join_data['cnt'] = join_data['be_flag'] + '|' + join_data['af_flag']
mm = pd.DataFrame().from_dict(dict(Counter(join_data['cnt'])), orient='index').reset_index()
mm.columns = ['name', 'cnt']
mm['before_sale'] = mm['name'].apply(lambda x: x.split('|')[0])
mm['after_sale'] = mm['name'].apply(lambda x: x.split('|')[1])
sns_data = mm.pivot("before_sale", "after_sale", "cnt").fillna(0.0)

fig = plt.figure()
ax = fig.add_subplot(figsize=(14, 8))
# sns.heatmap(mm.pivot("before_sale", "after_sale", "cnt").fillna(0.0), annot=True, fmt="d", linewidths=.5, cmap='YlGnBu', ax=ax)
sns.heatmap(sns_data, annot=True, fmt=".0f", linewidths=.5, cmap='YlGnBu', ax=ax)
plt.title('Sales migration')
plt.show()


# plot series of kpi
data = [[4 , 20641 , 1.2334488903751435,1.297406426703958,1.0169647169696603,1.0096910135587038,0.9972456983036421],
[5 , 14333 , 1.1720888179507842,1.221004717039609,0.9603272515961072,0.9506811256430916,0.9958191284943905],
[6 , 10329 , 1.1239744471047821,1.158306445718606,0.8956865855963708,0.8752276258584837,0.9953878142524097],
[7 , 7638 ,1.0900513154525668,1.1137926737933692,0.8573045480732303,0.9534541063884264,0.9950518499551919],
[9 , 4478 ,1.0469679130434786,1.058932461538462,0.8108711304347829,0.936437725752509,0.9930836120401337],
[10 , 3540 ,1.0314996979569673,1.0421080737120754,0.7780938641145728,0.7793817314871376,0.991905925473427],
[13 , 1979,1.0061014545196536,1.01060586774514,0.7160779409677874,0.724736274301121,0.988888888888889],
[15 , 1000 ,0.9969654363636352,0.9992984218181815,0.6519841672727268,0.7261242618181817,0.988054545454546]]
name = ['order' , 'cnt',     'wma_mape',      'hw_mape',   'reg_single_mape',    'reg_sum_mape',      'ma_mape']
ser_pd = pd.DataFrame(data, columns=name)

ser_pd.loc[:, ['order', 'cnt', 'reg_single_mape']]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ser_pd['order'], ser_pd['reg_single_mape'], label='reg_single_mape')
# ax.plot(ser_pd['order'], ser_pd['reg_sum_mape'], label='reg_sum_mape')
ax.set_title('Reg_single Mape (cnt change)')
ax.set_xlabel('cnt')
ax.set_ylabel('reg_single_mape')
plt.show()
