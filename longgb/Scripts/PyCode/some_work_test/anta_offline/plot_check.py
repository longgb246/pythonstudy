# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/8
  Usage   : 
"""

import pandas as pd
import os
import datetime
from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY

import matplotlib

matplotlib.use('TkAgg')  # 使用tk画图

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('seaborn-darkgrid')

start_date = '2018-08-11'
# start_date = '2018-07-30'
day_len = 90
conf = {
    'sale_table': 'sale_sum_08_11.xls',
    # 'dev_table': 'dev_models_08_07.xls',
    'dev_table': 'dev_models_08_11.xls',
    # 'online_table': 'online_models_08_07.xls',
    'online_table': 'online_models_08_11.xls',
    'combine_table': 'combine_models_08_09.xls',
    'dev2_table': 'dev2_models_07_31.xls',
    'data_list': ['dev_table', 'online_table'],
    # 'data_list': ['combine_table'],
    'name_list': ['dev', 'online'],
    # 'name_list': ['combine'],
}


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


def read_xls_csv(path):
    if path.endswith('csv'):
        df = pd.read_table(path)
    else:
        df = pd.read_excel(path)
    return df


def get_data(start_date, day_len, conf=conf):
    sale_table = conf['sale_table']
    data_list_name = conf['data_list']

    path = '/Users/longguangbin/Work/scripts/anta_offline'
    date_list = get_date_list(start_date, day_len)

    sale_sum = read_xls_csv(path + os.sep + sale_table)
    sale_sum.columns = ['sale', 'dt']

    data_list = []
    for each in data_list_name:
        df = read_xls_csv(path + os.sep + conf[each])
        df['dt'] = date_list
        df.columns = ['sale', 'dt']
        data_list.append(df)

    return sale_sum, data_list


def plot_func(start_date, before_day=30, sale_sum=None, data_list=None, name_list=None):
    before_date = (parse(start_date) - datetime.timedelta(before_day)).strftime('%Y-%m-%d')
    end_date = max(map(lambda x: np.max(x['dt']), data_list))
    week_df = get_week_df(before_date, end_date)
    tmp_sale = sale_sum[sale_sum['dt'] > before_date]
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(tmp_sale['dt'], tmp_sale['sale'])
    for i, each in enumerate(data_list):
        ax.plot(each['dt'], each['sale'], label=name_list[i])
    x_tick_labels = list(ax.get_xticklabels())
    tick_num = 10  # 刻度数目
    tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    # print x_labels, tick_spacing
    y_max = max([np.max(tmp_sale['sale'])] + map(lambda x: np.max(x['sale']), data_list))
    y_min = min([np.min(tmp_sale['sale'])] + map(lambda x: np.min(x['sale']), data_list))
    y_gap = y_max - y_min
    width = 1
    ax.bar(week_df['dt'], week_df['week'].apply(lambda x: y_max + y_gap * 0.2 if x == '1' else 0), width, color="red",
           align='center', alpha=0.25)
    # ax.yaxis.grid(False)
    ax.set_ylim(y_min - y_gap * 0.03, y_max + y_gap * 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.55))
    ax.set_title('Predict date : {0} . History Windows : {1} days.'.format(start_date, before_day), fontsize=15)
    plt.show()


sale_sum, data_list = get_data(start_date, day_len, conf=conf)
plot_func(start_date, before_day=90, sale_sum=sale_sum, data_list=data_list,
          name_list=conf['name_list'])


# ax.set_yticks(y_pos)                        # 设置y轴的刻度范围，原来是[-1,5]，从-1到5，设置后是从0-4， 与 ax.set_yticks([0, 4]) 等价
# ax.set_yticklabels(people)                  # 设置y轴的刻度标识


def get_hive_data():
    sale_sum_sql = '''
    select sum(sale) as sale, sale_date as dt 
    from app.app_saas_sfs_model_input
    where dt = date_add(CURRENT_DATE, -1)
    group by sale_date
    '''
    pre_sql = '''
    select 
        demand_sum 
    from 
        (
            select
                array(
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[0] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[1] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[2] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[3] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[4] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[5] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[6] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[7] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[8] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[9] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[10] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[11] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[12] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[13] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[14] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[15] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[16] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[17] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[18] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[19] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[20] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[21] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[22] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[23] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[24] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[25] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[26] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[27] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[28] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[29] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[30] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[31] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[32] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[33] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[34] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[35] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[36] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[37] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[38] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[39] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[40] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[41] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[42] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[43] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[44] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[45] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[46] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[47] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[48] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[49] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[50] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[51] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[52] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[53] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[54] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[55] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[56] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[57] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[58] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[59] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[60] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[61] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[62] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[63] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[64] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[65] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[66] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[67] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[68] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[69] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[70] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[71] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[72] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[73] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[74] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[75] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[76] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[77] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[78] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[79] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[80] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[81] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[82] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[83] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[84] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[85] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[86] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[87] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[88] as double)), 2),
                    round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[89] as double)), 2)
                ) as demand
            from 
                (
                    select 
                        sale_list
                    from 
                        app.dev_lgb_test_saas_sfs_rst 
                    where 
                        dt = date_add(CURRENT_DATE, -1)
                )   a
        )   b
    lateral view explode(b.demand) 
          c as demand_sum
    '''
    # app_saas_sfs_rst | app_lgb_test_bca_forecast_result_try
    pass


# Check the anta 的预测问题
# 1、check 预测输入+输出的 sku 个数

'''
-- have a look
select * from app.app_saas_sfs_model_input where dt = date_add(CURRENT_DATE, -1)
select * from app.app_saas_sfs_rst where dt = date_add(CURRENT_DATE, -1)
select * from app.app_lgb_test_bca_forecast_result where dt = date_add(CURRENT_DATE, -1)

-- input
app.app_saas_sfs_model_input
-- online
app.app_saas_sfs_rst 
-- dev
app.app_lgb_test_bca_forecast_result 

-- 检查预测输入与输出：sku 数量一致
-- 181846
select  count(sku_id) as cnt from 
(select distinct 
concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1'), case when length(sale_date) > 10 then concat('_', substring(sale_date, 12)) else '' end) as sku_id
from app.app_saas_sfs_model_input where dt = date_add(CURRENT_DATE, -1)
) a
-- 181846
select count(sku_id) as cnt from 
(select distinct 
concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1')) as sku_id
from  app.app_saas_sfs_rst where dt = date_add(CURRENT_DATE, -1)
) a
-- 181846
select count(distinct sku_id) from app.app_lgb_test_bca_forecast_result where dt = date_add(CURRENT_DATE, -1)
-- 181846
select count(sku_id) from app.app_lgb_test_bca_forecast_result where dt = date_add(CURRENT_DATE, -1)

'''
