# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@jd.com
  Date    : 2018/8/16
  Usage   : 
"""

import os
from dateutil.parser import parse
import datetime
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 使用tk画图

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dateutil.rrule import rrule, DAILY

dt = '2018-08-01'
path = '/Users/longguangbin/Work/scripts/anta_offline/detail'
sale_file = 'sale_active.tsv'
pre_file = 'pred_{0}.tsv'.format(dt)

# sku_code = '11721360-7/8.5'
sku_code = '15821204-3/2XL'
# store_id = 'L611'
store_id = 'KL00'  # '__all__'
models = ['reg_single', 'hw', 'wma', 'combine']  # reg_single | hw | wma | combine
pre_len = 7
before_day = 180


def get_date_range1(date_start, pre_len):
    date_start_dt = parse(date_start)
    return [(date_start_dt + datetime.timedelta(x + 1)).strftime('%Y-%m-%d') for x in range(pre_len)]


def get_all_date(date_begin, date_end):
    date_begin_dt = parse(date_begin)
    date_end_dt = parse(date_end)
    date_len = (date_end_dt - date_begin_dt).days + 1
    return [(date_begin_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range(date_len)]




def mm(l):
    l_list = map(lambda x: eval(x), l.values)
    return np.sum(l_list, axis=0).tolist()



def get_data(dt, path, sale_file, pre_file, sku_code, store_id, pre_len, models):
    pre_data = pd.read_table(path + os.sep + pre_file)
    real_data = pd.read_table(path + os.sep + sale_file)

    group_store = store_id.lower() == '__all__'

    date_range = get_date_range1(dt, pre_len)

    tmp_real = real_data[(real_data['sku_code'].apply(lambda x: sku_code in x)) & (
        real_data['store_id'].apply(lambda x: True if group_store else store_id == x))]
    tmp_real = tmp_real.groupby(['sku_code', 'dt']).agg({'qty': 'sum', 'sale': 'sum'}).reset_index()

    dt_min, dt_max = min([np.min(tmp_real['dt'])] + date_range), max([np.max(tmp_real['dt'])] + date_range)
    all_date_range = get_all_date(dt_min, dt_max)
    date_range_df = pd.DataFrame(all_date_range, columns=['dt'])

    pre_list = []
    for model in models:
        tmp_pre = pre_data[(pre_data['sku_code'].apply(lambda x: sku_code in x)) & (
            pre_data['store_id'].apply(lambda x: True if group_store else store_id == x)) & (
                               pre_data['sale_type'].apply(lambda x: model == x))]
        tmp_pre = tmp_pre.groupby(['sku_code']).agg(
            {'sale_list': lambda y: str(np.sum(map(lambda x: eval(x), y.values), axis=0).tolist())}).reset_index()
        tmp_value = tmp_pre['sale_list'].values
        if len(tmp_value) != 1:
            print(tmp_pre)
            raise Exception()
        else:
            pre_values = eval(tmp_value[0])[:pre_len]
            pre_df = pd.DataFrame(pre_values, columns=['sale'])
            pre_df['dt'] = date_range
        pre_list.append(pre_df)

    real_sale_df = date_range_df.merge(tmp_real.loc[:, ['sale', 'qty', 'dt']], on=['dt'], how='left').fillna(0)

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


def plot_func(start_date, before_day=30, real_sale=None, data_list=None, name_list=None, qty=True, dt_min=None,
              dt_max=None, sku_code=None, store_id=None):
    before_date = (parse(start_date) - datetime.timedelta(before_day)).strftime('%Y-%m-%d')
    dt_min_min = before_date
    week_df = get_week_df(dt_min_min, dt_max)
    tmp_sale = real_sale[real_sale['dt'] > dt_min_min]
    all_date_range = get_all_date(dt_min_min, dt_max)
    date_range_df = pd.DataFrame(all_date_range, columns=['dt'])
    real_sale = date_range_df.merge(tmp_sale.loc[:, ['sale', 'qty', 'dt']], on=['dt'], how='left').fillna(0)
    # tmp_sale = sale_sum[sale_sum['dt'] > before_date]
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    if qty:
        ax.plot(real_sale['dt'], real_sale['qty'], label='qty', alpha=0.9)
    ax.plot(real_sale['dt'], real_sale['sale'], label='real')
    name_list = ['dev'] * len(data_list) if name_list is None else name_list
    for i, each in enumerate(data_list):
        ax.plot(each['dt'], each['sale'], label=name_list[i])
    x_tick_labels = list(ax.get_xticklabels())
    tick_num = 10  # 刻度数目
    tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    # print x_labels, tick_spacing
    y_max = max([np.max(real_sale['sale'])] + map(lambda x: np.max(x['sale']), data_list))
    y_min = min([np.min(real_sale['sale'])] + map(lambda x: np.min(x['sale']), data_list))
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
          qty=True, dt_min=dt_min, dt_max=dt_max, sku_code=sku_code, store_id=store_id)
