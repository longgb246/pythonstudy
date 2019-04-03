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
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文
# mpl.rcParams['axes.unicode_minus'] = False  # 解决负号问题

from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/Users/longguangbin/STHeiti.ttc')
ktfont = FontProperties(fname='/Users/longguangbin/hwKaiti.ttf')

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


def columns_up(df):
    cols = [x.capitalize() for x in df.columns]
    df.columns = cols
    return df


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

    train_x = columns_up(train_x)
    train_y = columns_up(train_y)

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

    train_x = columns_up(train_x)
    train_y = columns_up(train_y)
    train_pd = columns_up(train_pd)

    return train_x, train_y, train_pd


def get_table1(statis_pd):
    tmp1 = statis_pd.groupby(['Fin_flag', 'Pass']).agg({'Fin_flag': 'count'}).to_dict()['Fin_flag']
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
    cols = ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust']
    add_table = []
    for col in cols:
        tmp = statis_pd.groupby([col, 'Pass']).agg({col: 'count'}).to_dict()[col]
        num_pass = tmp.get((1, 1), 0)
        num_no_pass = tmp.get((1, 0), 0)
        nums = num_pass + num_no_pass
        add_table.append([col, nums, num_pass, num_no_pass, '{0:.2f}%'.format(num_pass * 100.0 / nums)])

    table2 = table1[:2] + add_table + table1[2:]
    asc_table = AsciiTable(table2)
    print(asc_table.table)
    return asc_table


def get_table3(train_x):
    summary_cols = ['Finance', 'Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                    'Science', 'Law', 'Consult', 'Government',
                    'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                    'Soe', 'Sti']

    func_map = {'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max, 'std': np.std,
                'skew': stats.skew, 'kurtosis': stats.kurtosis}

    func_list = ['mean', 'median', 'min', 'max', 'std', 'skew', 'kurtosis']

    table3 = [[''] + summary_cols]
    for func in func_list:
        this_func = func_map.get(func)
        tmp = this_func(train_x.loc[:, summary_cols], axis=0).round(2).tolist()
        table3.append([func] + tmp)
    table3 = map(list, zip(*table3))

    asc_table = AsciiTable(table3)
    print(asc_table.table)
    return asc_table


def box_plot(data, diff_color=False, xlabeln='', ylabeln='', titlen='', labels=[], is_save=False, save_path=''):
    # plt.style.use('tableau-colorblind10')
    # plt.style.use('seaborn-darkgrid')
    # fig = plt.figure(figsize=size)

    if isinstance(data, pd.DataFrame):
        labels = data.columns if not labels else labels
        data = data.values.T.tolist()
    elif isinstance(data, pd.Series):
        labels = [data.name] if not labels else labels
        data = data.values.T.tolist()
    else:
        labels = ['x{0}'.format(i) for i in range(len(data))] if not labels else labels
        if isinstance(data, np.ndarray):
            data = data.T.tolist()

    cmap1 = cm.get_cmap('hsv')
    x = np.linspace(0.0, 1.0, 20)
    colors = cmap1(x).tolist() * 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bplot = ax.boxplot(data,
                       vert=False,
                       showmeans=True,  # 显示均值
                       meanline=True,  # 均值使用线
                       patch_artist=True,
                       boxprops=dict(linewidth=2, alpha=0.35),
                       )

    if diff_color:
        alpha = 0.35
        for i, patch in enumerate(bplot['boxes']):
            color = colors[i * 2 + 1][:-1] + [alpha]
            patch.set_facecolor(color)
            # 异常值
            flier = bplot['fliers'][i]
            flier.set_markeredgecolor('#A1A1A1')
            flier.set_markerfacecolor(color)

    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabeln)
    ax.set_ylabel(ylabeln)
    ax.set_title(titlen)
    if is_save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return [fig, ax]


def pair_plot(data, vars=[], hue='', is_save=False, save_path=''):
    sns.pairplot(data,
                 vars=vars,
                 hue=hue,
                 palette={1: '#BCD8F2', 0: 'r'},
                 # markers=["o", "D"],
                 markers=["D", "o"],
                 # kind="reg",
                 diag_kind="kde",
                 plot_kws=dict(edgecolor=None, alpha=0.5, linewidth=0),
                 diag_kws=dict(shade=True),
                 )
    if is_save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_stars(p):
    if p <= 0.01:
        return '***'
    elif p <= 0.05:
        return '**'
    elif p <= 0.1:
        return '*'
    else:
        return ''


def get_mix_model(train_y, train_x, cols=[]):
    # cols = ['a', 'B']
    keep_cols = [x for x in train_x.columns if x not in cols]
    table = [[''] + reduce(lambda m, n: m + n, map(lambda x: [x, ''], train_x.columns)) + \
             ['N', 'Adj.R-squared', 'F-statistic', 'AIC', 'BIC']]

    # 每个变量
    for i, col in enumerate(cols):
        tmp_cols = [''] * (len(cols) * 2)
        this_cols = [col] + keep_cols
        model = sm.OLS(train_y, train_x.loc[:, this_cols]).fit()

        coef_list = model.params.values.round(3).tolist()
        tvalues_list = model.tvalues.values.round(2).tolist()
        t_pvalues_list = [float(model.t_test(x).pvalue) for x in this_cols]

        tmp_cols[i * 2] = coef_list[0]
        tmp_cols[i * 2 + 1] = '({0})'.format(tvalues_list[0]) + get_stars(t_pvalues_list[0])

        for j in range(len(coef_list[1:])):
            tmp_cols.extend([coef_list[j], '({0})'.format(tvalues_list[j]) + get_stars(t_pvalues_list[j])])

        tmp_cols.extend([int(model.nobs), np.round(model.rsquared, 3),
                         str(np.round(model.fvalue, 2)) + get_stars(model.f_pvalue),
                         np.round(model.aic, 2), np.round(model.bic, 2)])

        table.append(['({0})'.format(i + 1)] + tmp_cols)

    # 所有变量一起
    model = sm.OLS(train_y, train_x).fit()

    coef_list = model.params.values.round(3).tolist()
    tvalues_list = model.tvalues.values.round(2).tolist()
    t_pvalues_list = [float(model.t_test(x).pvalue) for x in train_x.columns]

    tmp_cols = []
    for j in range(len(coef_list)):
        tmp_cols.extend([coef_list[j], '({0})'.format(tvalues_list[j]) + get_stars(t_pvalues_list[j])])

    tmp_cols.extend([int(model.nobs), np.round(model.rsquared, 3),
                     str(np.round(model.fvalue, 2)) + get_stars(model.f_pvalue),
                     np.round(model.aic, 2), np.round(model.bic, 2)])
    table.append(['({0})'.format(len(cols) + 1)] + tmp_cols)

    table = map(list, zip(*table))
    asc_table = AsciiTable(table)

    for i in range(len(cols) + 2):
        asc_table.justify_columns[i] = 'right'

    print(asc_table.table)
    return asc_table


def col2int(df, cols):
    for col in cols:
        df[col] = df[col].astype(int)
    return df


def plot_bar1(x, y, is_save=False, save_file=''):
    """ 统计总量 bar 图 """
    fig = plt.figure(figsize=(14, 8))

    ax = fig.add_subplot(111)

    ax.bar(x, y)
    ax.set_xlabel(u'年份', fontproperties=ktfont, size=14, labelpad=10)
    ax.set_title(u'每年申请可转债公司数目', fontproperties=ktfont, size=16, pad=10)
    for x1, y1 in zip(range(len(x)), y):
        ax.text(x1, y1 + 0.05, '{0}'.format(y1), ha='center', va='bottom',
                color='#4F94CD', fontsize=10)

    if is_save:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


def plot_bar2(cnt_pd, cnt_pd_pass, table=True, is_save=False, save_file=''):
    """ 画分开的通过、未通过、总量的bar图 """
    ind = np.arange(len(cnt_pd_pass))  # the x locations for the groups
    width = 0.45 if not table else 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))

    rects1 = ax.bar(ind - width / 2, cnt_pd_pass['pass'], width, color='SkyBlue', label=u'通过')
    rects2 = ax.bar(ind + width / 2, cnt_pd_pass['nopass'], width, color='IndianRed', label=u'未通过')

    ax.plot(ind, cnt_pd['pass'], label=u'总申请', color='r', alpha=0.6)

    def autolabel(rects, xpos='center'):
        """ Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.3, 'left': 0.85}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{0:.0f}'.format(height), ha=ha[xpos], va='bottom', color='#4F94CD')

    autolabel(rects1, "left")
    autolabel(rects2, "right")

    if table:
        rows = [u'通过', u'未通过', u'总申请']
        columns = cnt_pd['d_year'].values
        cell_text = [cnt_pd_pass['pass'].values, cnt_pd_pass['nopass'].values, cnt_pd['pass'].values]
        colors = ['SkyBlue', 'IndianRed', '#F6B2B2']

        # Add a table at the bottom of the axesax
        the_table = ax.table(cellText=cell_text,
                             rowLabels=rows,
                             rowColours=colors,
                             colLabels=columns,
                             loc='bottom')

        cellDict = the_table.get_celld()
        for key, value in cellDict.items():
            value.set_height(0.05)
            if key[1] == -1:
                value.set_text_props(fontproperties=ktfont)

        ax.set_xlim([-0.5, len(cnt_pd_pass['nopass'].values) - 0.5])
        ax.set_xticks([])

        ax.legend(prop=ktfont, framealpha=0.75, loc='upper left', bbox_to_anchor=(1, 0.55))
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2)

    else:
        ax.set_xlabel(u'年 份', fontproperties=ktfont, size=14, labelpad=10)
        ax.set_xticks(ind)
        ax.set_xticklabels(cnt_pd_pass['d_year'])
        ax.legend(prop=ktfont, framealpha=0.75, loc='upper right')

    ax.set_title(u'每年申请可转债(通过与未通过)公司数目', fontproperties=ktfont, size=16, pad=10)

    if is_save:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


def main():
    data_path = r'/Users/longguangbin/tmp/yuhan/financial_data'
    control_1, control_2, self_var_time, self_var_during = read_data(data_path)
    control_1 = make_factor(control_1)

    # --------- discriminant_model ---------
    train_x, train_y = discriminant_model_data(control_1, control_2, self_var_time)

    train_x = indicate(train_x, ['Finance', 'Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                 'Science', 'Law', 'Consult', 'Government'])
    org_train_x = train_x.copy()
    train_x = filter_quantile(train_x, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x = scale(train_x, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    # train_x = normalize(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 标准化

    train_fin_x = train_x.loc[:, ['Finance',
                                  'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                  'Soe', 'Sti']]
    train_fin_xs = train_x.loc[:, ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                   'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                   'Soe', 'Sti']]

    # --------- regression_model ---------
    train_x2, train_y2, train_pd2 = regression_model_data(control_1, control_2, self_var_during)

    train_x2 = indicate(train_x2, ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                   'Science', 'Law', 'Consult', 'Government', 'Finance'])
    train_x2 = filter_quantile(train_x2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x2 = scale(train_x2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    # train_x = normalize(train_x, ['asset', 'roa', 'cur', 'dar', 'ginc'])  # 标准化

    train_y2 = scale(train_y2, ['Y2'])

    train_fin_x2 = train_x2.loc[:, ['Finance',
                                    'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                    'Soe', 'Sti']]
    train_fin_xs2 = train_x2.loc[:, ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                     'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                     'Soe', 'Sti']]

    # --------- 稳健性预测 ---------
    train_x_w, train_y_w = discriminant_model_data(control_1, control_2, self_var_time)
    train_x_w = filter_quantile(train_x_w, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x_w = scale(train_x_w, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    train_fin_x_w = train_x_w.loc[:, ['Finance',
                                      'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                      'Soe', 'Sti']]
    train_fin_xs_w = train_x_w.loc[:, ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                       'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                       'Soe', 'Sti']]

    train_x_w2, train_y_w2, train_pd_w2 = regression_model_data(control_1, control_2, self_var_during)
    train_x_w2 = filter_quantile(train_x_w2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x_w2 = scale(train_x_w2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    train_y_w2 = scale(train_y_w2, ['Y2'])
    train_fin_x_w2 = train_x_w2.loc[:, ['Finance',
                                        'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                        'Soe', 'Sti']]
    train_fin_xs_w2 = train_x_w2.loc[:, ['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                         'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                         'Soe', 'Sti']]

    model = discriminant_model(train_y_w, train_fin_x_w)
    # model = discriminant_model(train_y_w, train_fin_xs_w)
    model = discriminant_model(train_y_w2, train_fin_x_w2)
    # model = discriminant_model(train_y_w2, train_fin_xs_w2)
    table4 = get_mix_model(train_y_w, train_fin_xs_w, cols=['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust'])
    table5 = get_mix_model(train_y_w2, train_fin_xs_w2, cols=['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust'])

    # 增加其他背景分析
    train_x_2w, train_y_2w = discriminant_model_data(control_1, control_2, self_var_time)
    train_x_2w = indicate(train_x_2w, ['Finance', 'Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust',
                                       'Science', 'Law', 'Consult', 'Government'])
    train_x_2w = filter_quantile(train_x_2w, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x_2w = scale(train_x_2w, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    train_fin_x_2w = train_x_2w.loc[:, ['Finance', 'Science', 'Law', 'Consult', 'Government',
                                        'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                        'Soe', 'Sti']]
    model = discriminant_model(train_y_2w, train_fin_x_2w)

    train_x_2w2, train_y_2w2, train_pd_2w2 = regression_model_data(control_1, control_2, self_var_during)
    train_x_2w2 = filter_quantile(train_x_2w2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])
    train_x_2w2 = scale(train_x_2w2, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'])  # 0-1规范化
    train_y_2w2 = scale(train_y_2w2, ['Y2'])
    train_fin_x_2w2 = train_x_2w2.loc[:, ['Finance', 'Science', 'Law', 'Consult', 'Government',
                                          'Asset', 'Roa', 'Cur', 'Dar', 'Ginc',
                                          'Soe', 'Sti']]
    model = discriminant_model(train_y_2w2, train_fin_x_2w2)

    # ------------------ simple statistics ------------------
    statis_pd = self_var_time.copy().fillna(0)
    statis_pd['finance'] = (statis_pd['bank5'] + statis_pd['bank_no5'] + statis_pd['broker'] +
                            statis_pd['insurance'] + statis_pd['trust'])
    statis_pd['fin_flag'] = statis_pd['finance'].apply(lambda x: '1' if x > 0 else '0')
    statis_pd = columns_up(statis_pd)

    # ------------------ check the data ------------------
    table1 = get_table1(statis_pd)
    table2 = get_table2(table1, statis_pd)
    table3 = get_table3(train_x)

    model = discriminant_model(train_y, train_fin_x)
    model = discriminant_model(train_y, train_fin_xs)
    model = discriminant_model(train_y2, train_fin_x2)
    model = discriminant_model(train_y2, train_fin_xs2)

    # model = forward_selected(train_y, train_fin_x)

    def pca_test():
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_r = pca.fit(train_fin_x.loc[:, ['Finance', 'Asset', 'Roa', 'Cur', 'Dar', 'Ginc']]). \
            transform(train_fin_x.loc[:, ['Finance', 'Asset', 'Roa', 'Cur', 'Dar', 'Ginc']])
        pca_data = pd.concat([pd.DataFrame(X_r, columns=['PCA_x', 'PCA_y']), train_y], axis=1)

        colors = ['#BCD8F2', 'r', 'navy', 'turquoise', 'darkorange']
        colors = ['#EEEE00', 'r', 'navy', 'turquoise', 'darkorange']
        # colors = ['turquoise', 'r', 'navy', 'turquoise', 'darkorange']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, label in enumerate(['1', '0']):
            tmp_data = pca_data.query(''' Pass == '{0}' '''.format(label))
            ax.scatter(tmp_data['PCA_x'], tmp_data['PCA_y'], color=colors[i])

        pca3 = PCA(n_components=3)
        X_r3 = pca3.fit(train_fin_x.loc[:, ['Finance', 'Asset', 'Roa', 'Cur', 'Dar', 'Ginc']]). \
            transform(train_fin_x.loc[:, ['Finance', 'Asset', 'Roa', 'Cur', 'Dar', 'Ginc']])
        pca_data = pd.concat([pd.DataFrame(X_r3, columns=['PCA_x', 'PCA_y', 'PCA_z']), train_y], axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(['1', '0']):
            tmp_data = pca_data.query(''' Pass == '{0}' '''.format(label))
            ax.scatter(tmp_data['PCA_x'], tmp_data['PCA_y'], tmp_data['PCA_z'], color=colors[i])

    from sklearn import linear_model

    # X is the 10x10 Hilbert matrix
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)
    n_alphas = 50
    alphas = np.logspace(-5, -1, n_alphas)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

    # train_fin_x

    table4 = get_mix_model(train_y, train_fin_xs, cols=['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust'])
    table5 = get_mix_model(train_y2, train_fin_xs2, cols=['Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust'])

    # #6FB1EA
    pair_plot(pd.concat([train_x, train_y], axis=1),
              vars=['Asset', 'Roa', 'Cur', 'Dar', 'Ginc', 'Soe', 'Sti'],
              hue='Pass', is_save=True, save_path=data_path + os.sep + 'tmp1.jpg')
    pair_plot(pd.concat([train_x, train_y], axis=1),
              vars=['Finance', 'Bank5', 'Bank_no5', 'Broker', 'Insurance', 'Trust'],
              hue='Pass', is_save=True, save_path=data_path + os.sep + 'tmp2.jpg')
    pair_plot(pd.concat([train_x, train_y], axis=1),
              vars=['Finance', 'Science', 'Law', 'Consult', 'Government'],
              hue='Pass', is_save=True, save_path=data_path + os.sep + 'tmp3.jpg')
    pair_plot(pd.concat([train_x, train_y], axis=1),
              vars=['Asset', 'Roa', 'Cur', 'Dar', 'Ginc'],
              hue='Pass', is_save=True, save_path=data_path + os.sep + 'tmp1_1.jpg')

    box_plot(org_train_x.loc[:, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc', 'Soe', 'Sti', 'Finance']])
    box_plot(train_x.loc[:, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc', 'Soe', 'Sti', 'Finance']])
    box_plot(train_x.loc[:, ['Asset', 'Roa', 'Cur', 'Dar', 'Ginc']], diff_color=True,
             is_save=True, save_path=data_path + os.sep + 'tmp4.jpg')

    # 统计图
    tmp_df = self_var_time.loc[:, ['year', 'pass']].rename(columns={'year': 'd_year'})

    cnt_pd = tmp_df.groupby(['d_year']).agg({'pass': 'count'}).reset_index()

    pas_df = tmp_df[tmp_df['pass'] == 1].groupby(['d_year']).agg({'pass': 'count'}).reset_index()
    no_pas_df = tmp_df[tmp_df['pass'] == 0].groupby(['d_year']).agg({'pass': 'count'}). \
        reset_index().rename(columns={'pass': 'nopass'})
    cnt_pd_pass = pas_df.merge(no_pas_df, on=['d_year'], how='left').fillna(0)
    cnt_pd_pass = col2int(cnt_pd_pass, ['pass', 'nopass'])
    plot_bar2(cnt_pd, cnt_pd_pass, is_save=True, save_file=data_path + os.sep + 'figure2_2_split.jpg')

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
