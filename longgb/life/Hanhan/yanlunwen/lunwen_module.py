# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/3/17
"""  
Usage Of 'lunwen_module.py' : 
"""

from __future__ import print_function
import os
import datetime
from dateutil.parser import parse
import numpy as np
import pandas as pd
from scipy import stats
from terminaltables import AsciiTable

import warnings

warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 150)  # 150
pd.set_option('display.max_columns', 40)


def col2str(df, cols):
    for col in cols:
        df[col] = df[col].astype(str)
    return df


def filter_quantile(df, cols, quantile=0.99):
    for col in cols:
        upper_v = np.percentile(df[col], int(quantile * 100))
        lower_v = np.percentile(df[col], int((1 - quantile) * 100))
        df[col] = df[col].apply(lambda x: upper_v if x > upper_v else x)
        df[col] = df[col].apply(lambda x: lower_v if x < lower_v else x)
    return df


def normalize(df, cols):
    """ 标准化：减去 mean 除以 std """
    for col in cols:
        df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])
    return df


def scale(df, cols):
    """ 规范化到 0-1 之间 """
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def indicate(df, cols):
    """ 转为指示表量 """
    for col in cols:
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
    return df


def read_data(path):
    # 财务指标
    control_1 = pd.read_csv(path + os.sep + 'control_vars.csv', encoding='gbk', converters={'Comcd': str, 'year': str})
    # 国企、战略性新兴企业打标
    control_2 = pd.read_csv(path + os.sep + 'control_vars2.csv', converters={'code': str, 'year': str})

    self_var_time = pd.read_excel(path + os.sep + 'self_vars.xlsx',
                                  sheet_name='time', converters={'code': str, 'year': str})
    self_var_during = pd.read_excel(path + os.sep + 'self_vars.xlsx',
                                    sheet_name='during', converters={'code': str, 'year': str})

    return control_1, control_2, self_var_time, self_var_during


def make_factor(control_1):
    control_1 = control_1.sort_values(['Comcd', 'year'])

    control_1['asset'] = np.log(control_1['Totass'])
    control_1['roa'] = control_1['Totalprf'] / (
            (control_1['Totass'] + control_1.groupby(['Comcd'])['Totass'].shift(1).fillna(method='backfill')) / 2)
    control_1['cur'] = control_1['Totcurass'] / control_1['Totcurlia']
    control_1['dar'] = control_1['Totlia'] / control_1['Totass']
    control_1['ginc'] = (control_1['Incmope'] -
                         control_1.groupby(['Comcd'])['Incmope'].shift(1).fillna(method='backfill')) / \
                        (control_1.groupby(['Comcd'])['Incmope'].shift(1).fillna(method='backfill'))

    control_1['y2'] = control_1['Netprf'] / control_1['Totass']
    control_1['y2'] = control_1.groupby(['Comcd'])['y2'].rolling(window=3).mean().values

    return control_1


def discriminant_model_data(control_1, control_2, self_var_time):
    # 判断是否成功
    # 06年发行用05年的财报的数据，所以往前推一年
    control_1['j_year'] = control_1['year'].astype(str).apply(lambda x: str(int(x) + 1))

    join_pd = self_var_time.merge(control_1.loc[:, ['Comcd', 'asset', 'roa', 'cur', 'dar', 'ginc', 'j_year']],
                                  left_on=['year', 'code'], right_on=['j_year', 'Comcd']). \
        merge(control_2, on=['year', 'code'])
    train_pd = join_pd.fillna(0)

    train_x = train_pd.loc[:, ['bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                               'science', 'law', 'consult', 'government',
                               'asset', 'roa', 'cur', 'dar', 'ginc',
                               'soe', 'sti']]
    train_y = train_pd.loc[:, ['pass']]

    # # 打标处理
    # train_x['finance'] = (train_pd['bank5'] + train_pd['bank_no5'] + train_pd['broker'] +
    #                       train_pd['insurance'] + train_pd['trust']).apply(lambda x: 1 if x > 0 else 0)
    # 累加处理
    train_x['finance'] = (train_pd['bank5'] + train_pd['bank_no5'] + train_pd['broker'] +
                          train_pd['insurance'] + train_pd['trust'])
    # train_x['finance'] = train_y['pass']

    return train_x, train_y


def discriminant_model(train_y, train_x):
    # model = sm.Logit(train_y, train_x).fit()
    # print(model.summary())

    model = sm.OLS(train_y, train_x).fit()
    print(model.summary())

    # glsar_model = sm.GLSAR(train_y, train_x, 1)
    # glsar_results = glsar_model.iterative_fit(1)
    # print(glsar_results.summary())
    return model


def forward_selected(Y, X, n_iter=100):
    """Linear model designed by forward selection."""
    remaining = list(X.columns)
    selected = []
    this_iter = 0
    current_score, best_new_score = 0.0, 0.0

    while remaining and current_score == best_new_score and this_iter <= n_iter:
        scores_with_candidates = []
        for candidate in remaining:
            score = sm.OLS(Y, X.loc[:, selected + [candidate]]).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()

        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    model = sm.OLS(Y, X.loc[:, selected]).fit()
    print(model.summary())

    # 获取 t 值的变化
    # dir(model)
    # model.t_test('consult').pvalue
    # model.tvalues

    return model


def regression_model_data(control_1, control_2, self_var):
    # 判断是否成功
    # 06年发行用05年的财报的数据，所以往前推一年
    control_1['j_year'] = control_1['year'].astype(str).apply(lambda x: str(int(x) + 1))
    control_1['y_year'] = control_1['year'].astype(str).apply(lambda x: str(int(x) - 2))

    join_pd = self_var.merge(control_1.loc[:, ['Comcd', 'asset', 'roa', 'cur', 'dar', 'ginc', 'j_year']],
                             left_on=['year', 'code'], right_on=['j_year', 'Comcd']). \
        merge(control_2, on=['year', 'code'])
    train_pd = join_pd.fillna(0)

    train_y = self_var.loc[:, ['year', 'code']].merge(control_1.loc[:, ['Comcd', 'y_year', 'y2']],
                                                      left_on=['year', 'code'], right_on=['y_year', 'Comcd']).fillna(0)

    train_pd = train_pd.merge(train_y, on=['code', 'year'])

    train_x = train_pd.loc[:, ['bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                               'science', 'law', 'consult', 'government',
                               'asset', 'roa', 'cur', 'dar', 'ginc',
                               'soe', 'sti']]
    train_y = train_pd.loc[:, ['y2']]

    # # 打标处理
    # train_x['finance'] = (train_pd['bank5'] + train_pd['bank_no5'] + train_pd['broker'] +
    #                       train_pd['insurance'] + train_pd['trust']).apply(lambda x: 1 if x > 0 else 0)
    # 累加处理
    train_x['finance'] = (train_pd['bank5'] + train_pd['bank_no5'] + train_pd['broker'] +
                          train_pd['insurance'] + train_pd['trust'])
    # train_x['finance'] = train_y['pass']

    return train_x, train_y, train_pd


def get_table1(statis_pd):
    tmp1 = statis_pd.groupby(['fin_flag', 'pass']).agg({'fin_flag': 'count'}).to_dict()['fin_flag']
    num_fin_pass = tmp1[('1', 1)]
    num_fin_no_pass = tmp1[('1', 0)]
    num_fin = num_fin_pass + num_fin_no_pass
    num_no_fin_pass = tmp1[('0', 1)]
    num_no_fin_no_pass = tmp1[('0', 0)]
    num_no_fin = num_no_fin_pass + num_no_fin_no_pass

    table1 = [['', 'Number', 'Pass', 'NoPass', 'Percent'],
              ['Finance', num_fin, num_fin_pass, num_fin_no_pass,
               '{0:.2f}%'.format(num_fin_pass * 100.0 / num_fin)],
              ['No Finance', num_no_fin, num_no_fin_pass, num_no_fin_no_pass,
               '{0:.2f}%'.format(num_no_fin_pass * 100.0 / num_no_fin)],
              ['Total', num_fin + num_no_fin, num_fin_pass + num_no_fin_pass, num_fin_no_pass + num_no_fin_no_pass,
               '{0:.2f}%'.format((num_fin_pass + num_no_fin_pass) * 100.0 / (num_fin + num_no_fin))]]

    print(AsciiTable(table1).table)
    return table1


def get_table2(table1, statis_pd):
    cols = ['bank5', 'bank_no5', 'broker', 'insurance', 'trust']
    add_table = []
    for col in cols:
        tmp = statis_pd.groupby([col, 'pass']).agg({col: 'count'}).to_dict()[col]
        num_pass = tmp.get((1, 1), 0)
        num_no_pass = tmp.get((1, 0), 0)
        nums = num_pass + num_no_pass
        add_table.append([col, nums, num_pass, num_no_pass, '{0:.2f}%'.format(num_pass * 100.0 / nums)])

    table2 = table1[:2] + add_table + table1[2:]
    print(AsciiTable(table2).table)
    return table2


def get_table3(train_x):
    summary_cols = ['finance', 'bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                    'science', 'law', 'consult', 'government',
                    'asset', 'roa', 'cur', 'dar', 'ginc', 'soe', 'sti']

    func_map = {'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max, 'std': np.std,
                'skew': stats.skew, 'kurtosis': stats.kurtosis}

    func_list = ['mean', 'median', 'min', 'max', 'std', 'skew', 'kurtosis']

    table3 = [[''] + summary_cols]
    for func in func_list:
        this_func = func_map.get(func)
        tmp = this_func(train_x.loc[:, summary_cols], axis=0).round(2).tolist()
        table3.append([func] + tmp)
    table3 = map(list, zip(*table3))

    print(AsciiTable(table3).table)
    return table3


def main():
    data_path = r'/Users/longguangbin/tmp/yuhan/financial_data'
    control_1, control_2, self_var_time, self_var_during = read_data(data_path)
    control_1 = make_factor(control_1)

    # --------- discriminant_model ---------
    train_x, train_y = discriminant_model_data(control_1, control_2, self_var_time)

    train_x = indicate(train_x, ['finance', 'bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                                 'science', 'law', 'consult', 'government'])
    train_x = filter_quantile(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])
    train_x = scale(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 0-1规范化
    # train_x = normalize(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 标准化

    train_fin_x = train_x.loc[:, ['finance',
                                  'asset', 'roa', 'cur', 'dar', 'ginc',
                                  'soe', 'sti']]
    train_fin_xs = train_x.loc[:, ['bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                                   'asset', 'roa', 'cur', 'dar', 'ginc',
                                   'soe', 'sti']]

    # --------- regression_model ---------
    train_x2, train_y2, train_pd2 = regression_model_data(control_1, control_2, self_var_during)

    train_x2 = indicate(train_x2, ['bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                                   'science', 'law', 'consult', 'government'])
    train_x2 = filter_quantile(train_x2, ['asset', 'roa', 'cur', 'dar', 'ginc'])
    train_x2 = scale(train_x2, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 0-1规范化
    # train_x = normalize(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 标准化

    train_y2 = scale(train_y2, ['y2'])

    train_fin_x2 = train_x2.loc[:, ['finance',
                                    'asset', 'roa', 'cur', 'dar', 'ginc',
                                    'soe', 'sti']]
    train_fin_xs2 = train_x2.loc[:, ['bank5', 'bank_no5', 'broker', 'insurance', 'trust',
                                     'asset', 'roa', 'cur', 'dar', 'ginc',
                                     'soe', 'sti']]

    # --------- simple statistics ---------
    statis_pd = self_var_time.copy().fillna(0)
    statis_pd['finance'] = (statis_pd['bank5'] + statis_pd['bank_no5'] + statis_pd['broker'] +
                            statis_pd['insurance'] + statis_pd['trust'])
    statis_pd['fin_flag'] = statis_pd['finance'].apply(lambda x: '1' if x > 0 else '0')

    # ------------------ check the data ------------------
    table1 = get_table1(statis_pd)
    table2 = get_table2(table1, statis_pd)
    table3 = get_table3(train_x)

    model = discriminant_model(train_y, train_fin_x)
    model = discriminant_model(train_y, train_fin_xs)
    model = discriminant_model(train_y2, train_fin_x2)
    model = discriminant_model(train_y2, train_fin_xs2)

    # #6FB1EA
    g4 = sns.pairplot(pd.concat([train_x, train_y], axis=1),
                      vars=['asset', 'roa', 'cur', 'dar', 'ginc', 'soe', 'sti'],
                      hue="pass",
                      palette={1: '#BCD8F2', 0: 'r'},
                      # markers=["o", "D"],
                      markers=["D", "o"],
                      # kind="reg",
                      diag_kind="kde",
                      plot_kws=dict(edgecolor=None, alpha=0.5, linewidth=0),
                      diag_kws=dict(shade=True),
                      )
    plt.savefig(data_path + os.sep + 'tmp1.jpg')
    plt.close()

    g4 = sns.pairplot(pd.concat([train_x, train_y], axis=1),
                      vars=['finance', 'bank5', 'bank_no5', 'broker', 'insurance', 'trust'],
                      hue="pass",
                      palette={1: '#BCD8F2', 0: 'r'},
                      # markers=["o", "D"],
                      markers=["D", "o"],
                      # kind="reg",
                      diag_kind="kde",
                      plot_kws=dict(edgecolor=None, alpha=0.5, linewidth=0),
                      diag_kws=dict(shade=True),
                      )
    plt.savefig(data_path + os.sep + 'tmp2.jpg')
    plt.close()

    g4 = sns.pairplot(pd.concat([train_x, train_y], axis=1),
                      vars=['finance', 'science', 'law', 'consult', 'government'],
                      hue="pass",
                      palette={1: '#BCD8F2', 0: 'r'},
                      # markers=["o", "D"],
                      markers=["D", "o"],
                      # kind="reg",
                      diag_kind="kde",
                      plot_kws=dict(edgecolor=None, alpha=0.5, linewidth=0),
                      diag_kws=dict(shade=True),
                      )
    plt.savefig(data_path + os.sep + 'tmp3.jpg')
    plt.close()


def plotBoxPlot(data, size=(8, 8), diff_color=False, xlabeln='x', ylabeln='y', titlen='', xticklabels=[]):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)

    # boxplot的属性
    boxprops = dict(linewidth=2, facecolor='#4C72B0', alpha=0.35)  # 盒子属性
    whiskerprops = dict(linewidth=2.5, linestyle='--', color='#797979', alpha=0.8)  # 虚线条属性
    flierprops = dict(linewidth=2, marker='o', markerfacecolor='none', markersize=6, linestyle='none')  # 异常值
    medianprops = dict(linestyle='-', linewidth=2.5, color='#FFA455')  # 中位数
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='#C44E52')  # 均值
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='r', alpha=0.6)  # 均值
    capprops = dict(linestyle='-', linewidth=2.5, color='#797979', alpha=0.8)  # 边界横线
    bplot = ax.boxplot(data,
                       vert=True,  # vertical box aligmnent
                       showmeans=True,  # 显示均值
                       meanline=True,  # 均值使用线
                       patch_artist=True,  # fill with color
                       boxprops=boxprops,  # 盒子属性
                       whiskerprops=whiskerprops,  # 虚线条属性
                       capprops=capprops,  # 边界横线
                       flierprops=flierprops,  # 异常值
                       medianprops=medianprops,  # 中位数  #FFA455   #797979    #3E3E3E
                       meanprops=meanlineprops  # 异常值
                       )
    colors = ['pink', 'lightblue', 'lightgreen', '#6AB27B', '#a27712', '#8172B2', '#4C72B0', '#C44E52', '#FFA455',
              '#797979'] * 4

    # 添加 box 的颜色
    if diff_color:
        for patch, color in zip(bplot['boxes'], colors[:len(bplot['boxes'])]):
            patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data))], )

    if xticklabels == []:
        xticklabels = ['x{0}'.format(x) for x in range(1, len(data) + 1)]
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabeln)
    ax.set_ylabel(ylabeln)
    ax.set_title(titlen)
    return [fig, ax]


def plotBoxPlotDemo():
    import numpy as np
    data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    plotBoxPlot(data)
    pass

    # 统计图

    # plot_kws=dict(s=50, edgecolor="b", linewidth=1),

    # model = discriminant_model(train_y, train_fin_x)
    # # model.rsquared_adj
    # # 因为数据都标准化，且0-1之间，常数项影响太大，所以不加常数项
    # # discriminant_model(train_y, sm.add_constant(train_fin_x))
    # model = forward_selected(train_y, train_fin_x)
    # model = discriminant_model(train_y, train_x.drop(['finance'], axis=1))
    # model = forward_selected(train_y, train_x.drop(['finance'], axis=1))

    # model = discriminant_model(train_y, train_fin_x)
    # model = forward_selected(train_y, train_fin_x)
    # model = discriminant_model(train_y, train_x.drop(['finance'], axis=1))
    # model = forward_selected(train_y, train_x.drop(['finance'], axis=1))


if __name__ == '__main__':
    main()
