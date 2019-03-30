# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/3/27
"""  
Usage Of 'lunwen_statistics.py' : 
"""

from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import datetime
from dateutil.parser import parse
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号问题

from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/Users/longguangbin/STHeiti.ttc')
ktfont = FontProperties(fname='/Users/longguangbin/hwKaiti.ttf')

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 150)  # 150
pd.set_option('display.max_columns', 40)


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


def no_use1():
    org_data_path = r'/Users/longguangbin/tmp/yuhan/org_data'
    results = pd.read_csv(org_data_path + os.sep + 'results.csv', encoding='gbk')
    results['d_year'] = results['year'].apply(lambda x: parse(x).strftime('%Y'))

    results['type'].drop_duplicates()
    type_list = [u'可转债', u'可换债', u'可转换公司债券', u'可转换债券']

    # 只提取可转债
    need_results = results[results['type'].apply(lambda x: x in type_list)]
    need_results['flag'] = need_results['pass'].apply(lambda x: '1' if x in [u'获通过', u'通过'] else '0')

    # 画总的申请数量的 bar 图
    cnt_pd = need_results.groupby(['d_year']).agg({'pass': 'count'}).reset_index()
    plot_bar1(cnt_pd['d_year'], cnt_pd['pass'], is_save=True,
              save_file=org_data_path + os.sep + 'figure1_total_bar.jpg')

    # 画分开的通过、未通过、总量的 bar 图
    cnt_pd1 = need_results.groupby(['d_year', 'flag']).agg({'pass': 'count'}).reset_index()
    cnt_pd_pass = cnt_pd1[cnt_pd1['flag'] == '1'].drop(['flag'], axis=1).merge(
        cnt_pd1[cnt_pd1['flag'] == '0'].rename(columns={'pass': 'nopass'}).drop(['flag'], axis=1),
        on=['d_year'], how='left').fillna(0)
    cnt_pd_pass = col2int(cnt_pd_pass, ['pass', 'nopass'])

    plot_bar2(cnt_pd, cnt_pd_pass, table=False, is_save=True, save_file=org_data_path + os.sep + 'figure2_1_split.jpg')
    plot_bar2(cnt_pd, cnt_pd_pass, is_save=True, save_file=org_data_path + os.sep + 'figure2_2_split.jpg')


def basic_statistics():
    # 统计数量
    data_path = r'/Users/longguangbin/tmp/yuhan/financial_data'
    self_var_time = pd.read_excel(data_path + os.sep + 'self_vars.xlsx',
                                  sheet_name='time', converters={'code': str, 'year': str})

    self_var_time = self_var_time.fillna(0)
    self_var_time['finance'] = (self_var_time['bank5'] + self_var_time['bank_no5'] + self_var_time['broker'] +
                                self_var_time['insurance'] + self_var_time['trust'])
    self_var_time['finance_flag'] = self_var_time['finance'].apply(lambda x: 1 if x > 0 else 0)

    mm = self_var_time.loc[:, ['code', 'flag', 'pass']].drop_duplicates()
    mm[mm['flag'] == 1].groupby(['pass']).agg({'flag': 'count'}).reset_index()
    mm[mm['flag'] == 0].groupby(['pass']).agg({'flag': 'count'}).reset_index()


def main():
    pass


if __name__ == '__main__':
    main()


# ==================================================================
def plot_bar_drop_no_use(x, y):
    """ 结论：axisartist 的中文显示会有问题，不建议使用！！！ """
    fig = plt.figure(figsize=(14, 8))
    ax_s = axisartist.Subplot(fig, 111)
    ax = fig.add_subplot(ax_s, 111)

    ax.bar(x, y)
    ax.set_title(u'每年申请可转债公司数目', fontproperties=font, size=16)
    ax.set_xlabel(u'年份', fontproperties=font, size=14)
    for x1, y1 in zip(range(len(x)), y):
        ax.text(x1, y1 + 0.05, '{0}'.format(y1), ha='center', va='bottom',
                color='#4F94CD', fontsize=10, fontweight='bold')

    # 通过 set_axisline_style 方法设置绘图区的底部及左侧坐标轴样式
    ax_s.axis["bottom"].set_axisline_style("->", size=1.5)  # "-|>"代表实心箭头："->"代表空心箭头
    ax_s.axis["left"].set_axisline_style("->", size=1.5)
    ax_s.axis['top'].set_visible(False)  # 去掉坐标
    ax_s.axis['right'].set_visible(False)

    plt.show()


def tmp_tmp():
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    recipe = ["225 g flour",
              "90 g sugar",
              "1 egg",
              "60 g butter",
              "100 ml milk",
              "1/2 package of yeast"]

    data = [225, 90, 50, 60, 100, 5]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title("Matplotlib bakery: A donut")

    plt.show()

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    size = 0.3
    vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
    # normalize vals to 2 pi
    valsnorm = vals / np.sum(vals) * 2 * np.pi
    # obtain the ordinates of the bar edges
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    ax.bar(x=valsleft[:, 0],
           width=valsnorm.sum(axis=1), bottom=1 - size, height=size,
           color=outer_colors, edgecolor='w', linewidth=1, align="edge")

    ax.bar(x=valsleft.flatten(),
           width=valsnorm.flatten(), bottom=1 - 2 * size, height=size,
           color=inner_colors, edgecolor='w', linewidth=1, align="edge")

    ax.set(title="Pie plot with `ax.bar` and polar coordinates")
    ax.set_axis_off()
    plt.show()

    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(vals.flatten(), radius=1 - size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    plt.show()
