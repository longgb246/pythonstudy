# -*- coding:utf-8 -*-
from __future__ import division
__author__ = 'xugang&tanxiao'
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from collections import Counter

report_path = '\\report'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_path(filePath):
    rst_path = filePath + report_path
    if os.path.exists(rst_path) == False:
        os.mkdir(rst_path)
    return rst_path


def plotkpi(kpi, filePath):
    # kpi = kpi_frame; filePath = analysis_path
    rst_path = get_path(filePath)

    # 周转天数图
    xlabeln = u'周转天数'
    ylabeln = u'频数'
    ylabel2n = u'周转天数累积分布'
    titlen = u'周转分布'
    fig, ax1 = plt.subplots()
    ret = plt.hist(kpi.TD, bins=50, range=[0.1, 200], color='#0070C0')  # 这里将一些特殊的sku去掉：为0的，和超过200的
    counts, bins, patches = ret[0], ret[1], ret[2]
    ax2 = ax1.twinx()
    sum_counts = np.cumsum(counts) / counts.sum()
    plt.plot(bins[1:], sum_counts, color='#C44E52')
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    ax2.set_ylabel(ylabel2n)
    ax1.set_ylim(-counts.max() * 0.05, counts.max() * 1.05)
    ax1.set_xlim(-10, 200 * 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.grid(False)
    plt.savefig(rst_path + '\\td')
    # 第二版
    binsn = range(0,100,10) + [np.inf]
    save_path = rst_path + '\\td2'
    plothistper(kpi[(kpi.TD.notnull())&(kpi.TD != np.inf)&(kpi.TD != -np.inf)]["TD"], binsn, xlabeln, ylabeln, titlen, save_path, cum_True=True)

    # 现货率图
    xlabeln = u'现货率'
    ylabeln = u'频数'
    ylabel2n = u'现货率累积分布'
    titlen = u'现货率分布'
    fig, ax1 = plt.subplots()
    ret = plt.hist(kpi.CR, bins=50, label='Z', color='#0070C0')
    counts, bins, patches = ret[0], ret[1], ret[2]
    ax2 = ax1.twinx()
    sum_counts = np.cumsum(counts) / counts.sum()
    plt.plot(bins[1:], sum_counts, color='#C44E52')
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    ax2.set_ylabel(ylabel2n)
    ax1.set_ylim(-counts.max() * 0.05, counts.max() * 1.05)
    ax1.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.grid(False)
    plt.savefig(rst_path+'\\cr')
    # 第二版
    binsn = np.linspace(0,1,11)
    save_path = rst_path + '\\cr2'
    plothistper(kpi[(kpi.CR.notnull())&(kpi.CR != np.inf)&(kpi.CR != -np.inf)]["CR"], binsn, xlabeln, ylabeln, titlen, save_path, cum_True=True, intshu=False)

    # 现货率和周转天数图
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    # ax.grid()
    ax.set_xlabel(u"周转天数")
    ax.set_ylabel(u"现货率")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    plt.scatter(kpi.TD, kpi.CR, color='#0070C0')
    plt.plot([60, 60], [0, 1], '--', color='red')
    plt.plot([0, 200], [0.8, 0.8], '--', color='red')
    plt.annotate('(1)', xy=(1, 1), xytext=(25, 0.9), fontsize=20, color='red')
    plt.annotate('(2)', xy=(1, 1), xytext=(130, 0.9), fontsize=20, color='red')
    plt.annotate('(3)', xy=(1, 1), xytext=(25, 0.4), fontsize=20, color='red')
    plt.annotate('(4)', xy=(1, 1), xytext=(130, 0.4), fontsize=20, color='red')
    plt.savefig(rst_path+'\\cr_td')


# 画出一个sku的库存与销量曲线
def plotsku(sku_data, kpi, filePath):
    id = sku_data.item_sku_id.unique()[0]

    # sku_data = sku
    # sku_data = sku_data.sort_values(u'日期')

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.set_xlabel(u'日期')
    ax1.set_ylabel(u'库存')
    ax2.set_ylabel(u'销量')
    ax2.yaxis.grid(False)
    # ax1.set_ylim(0, sku_data.stock_qtty.max() * 1.5)
    ax2.set_ylim(0, sku_data.total_sales.max() * 1.5)
    plt.title(u'sku:%d 周转:%.2f 现货率:%.2f' % (id, kpi.TD[id], kpi.CR[id]))
    ax1.plot(sku_data.day_string, sku_data.stock_qtty, '-', label=u'库存', color='#6AB27B') #6AB27B #E4D354
    ax2.plot(sku_data.day_string, sku_data.total_sales, '-', label=u'销量', color='#0070C0')#4C72B0 #91E8E1
    legend1 = ax1.legend(loc=(.89, .94), fontsize=12, shadow=True)
    legend2 = ax2.legend(loc=(.89, .9), fontsize=12, shadow=True)
    fname = filePath + '\\' + str(id)
    plt.savefig(fname)
    plt.close()


def plotZ(z_value_frame, filePath):
    rst_path = get_path(filePath)
    # rst_path = r'D:\Lgb\ReadData\RealityAnalysis\report'
    # save_path = r'D:\Lgb\ReadData\RealityAnalysis\report\tet'
    plothistper(z_value_frame.z_value, [-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf], u'z值', u'频数', u"Z值分布图", rst_path + '\\z')
    reta = plt.hist(z_value_frame.z_value, bins=[-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf], label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    retb = plt.hist(z_value_frame.z_value, bins=[-np.inf, 100, np.inf], label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    countsa, binsa, patchesa = reta[0], reta[1], reta[2]
    countsb, binsb, patchesb = retb[0], retb[1], retb[2]
    countsa_all = np.sum(countsa)
    z_output = [[u'Z值范围',u'[-Inf,0)',u'[0,2)',u'[2,4)',u'[4,6)',u'[6,8)',u'[8,10)',u'[10,12)',u'[12,14)',u'[14,Inf)',u'总数']]
    z_output.append([u'数量']+map(lambda x: unicode('{0:.0f}'.format(x)),np.append(countsa,countsa_all)))
    z_output.append([u'比例']+map(lambda x: unicode('{0:.2f}%'.format(x/countsa_all*100)),np.append(countsa,countsa_all)))
    z_output2 = u'\t1、Z值大于14的采购次数为{0:.0f}，Z值大于100的采购次数为{1:.0f}。这部分采购过早，造成高周转。\n\t2、Z值小于0的采购次数为{2:.0f}。这部分采购过晚，造成服务水平低。'.format(countsa[-1],countsb[1],countsa[0])
    # 画 BP 图
    # 1、原版
    fig2 = plt.figure()
    plt.xlabel(u'BP(天数)')
    plt.ylabel(u'频数')
    plt.hist([z_value_frame[(z_value_frame.band == 'L') | (z_value_frame.band == 'Z')].bp,
              z_value_frame[(z_value_frame.band == 'E') | (z_value_frame.band == 'F')].bp,
              z_value_frame[(z_value_frame.band == 'C') | (z_value_frame.band == 'D')].bp,
              z_value_frame[(z_value_frame.band == 'A') | (z_value_frame.band == 'B')].bp],
             label=['L/Z', 'E/F', 'C/D', 'A/B'], stacked=True)
    plt.legend(loc='upper right')
    plt.savefig(rst_path + '\\bp')
    # 参数
    xlabeln = u'BP范围'
    ylabeln = u'频数'
    titlen = u"BP值分布图"
    size = (12, 8)
    color_index = ["#4C72B0", "#6AB27B", "#C44E52", "#8172B2"]
    width = 0.5
    width2 = 0
    label_name_1 = ['L/Z', 'E/F', 'C/D', 'A/B']
    binsn = range(0,70,10) + [np.inf]
    # 2.1、第二版：全
    ret1 = plt.hist([z_value_frame[(z_value_frame.band == 'L') | (z_value_frame.band == 'Z')].bp,
              z_value_frame[(z_value_frame.band == 'E') | (z_value_frame.band == 'F')].bp,
              z_value_frame[(z_value_frame.band == 'C') | (z_value_frame.band == 'D')].bp,
              z_value_frame[(z_value_frame.band == 'A') | (z_value_frame.band == 'B')].bp],
             label=['L/Z', 'E/F', 'C/D', 'A/B'], bins=binsn)
    counts, bins, patches = ret1[0], ret1[1], ret1[2]
    counts_sum_1 = np.sum(counts, axis=0)
    fig1, ax1 = plt.subplots(figsize=size)
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x, bins)
    bins_name = ["[" + str(bins[i]) + "," + str(bins[i + 1]) + ")" for i in range(len(bins) - 1)]
    ind = np.arange(len(counts[0]))
    for i in range(len(counts)):
        ax1.bar(ind + width2, counts[i], width=width, color = color_index[i],
                bottom=np.sum(counts[:i], axis=0), tick_label=bins_name, align='center', label=label_name_1[i])
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
    plt.title(titlen)
    plt.savefig(rst_path + '\\bp2')
    print "BP1: ",
    for each in counts_sum_1:
        print "{0:7.0f} | ".format(each),
    print " total: {0}".format(np.sum(counts_sum_1))
    print "     ",
    for each in counts_sum_1:
        print "{0:6.2f}% | ".format(each/np.sum(counts_sum_1)*100),
    print " total: 100%"
    print "大于{0:.0f}天的sku占{1:.2f}%。".format(binsn[3], np.sum(counts_sum_1[3:])/np.sum(counts_sum_1)*100)
    # table
    bp1_output = [[u'BP值范围',u'小于10天',u'10-20天',u'20-30天',u'30-40天',u'40-50天',u'50-60天',u'大于60天',u'总数']]
    bp1_output.append([u'数量']+map(lambda x: unicode('{0:.0f}'.format(x)),np.append(counts_sum_1,np.sum(counts_sum_1))))
    bp1_output.append([u'比例']+map(lambda x: unicode('{0:.2f}%'.format(x/np.sum(counts_sum_1)*100)),np.append(counts_sum_1,np.sum(counts_sum_1))))
    bp1_output2 = u"大于{0:.0f}天的sku占{1:.2f}%。".format(binsn[3], np.sum(counts_sum_1[3:])/np.sum(counts_sum_1)*100)
    # 2.2、第三版：仅ABCD
    label_name_2 = ['A', 'B', 'C', 'D']
    ret2 = plt.hist([z_value_frame[(z_value_frame.band == 'A')].bp,
              z_value_frame[(z_value_frame.band == 'B')].bp,
              z_value_frame[(z_value_frame.band == 'C')].bp,
              z_value_frame[(z_value_frame.band == 'D')].bp],
             label=['A', 'B', 'C', 'D'], bins=binsn)
    counts, bins, patches = ret2[0], ret2[1], ret2[2]
    counts_sum_2 = np.sum(counts, axis=0)
    fig1, ax1 = plt.subplots(figsize=size)
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x, bins)
    bins_name = ["[" + str(bins[i]) + "," + str(bins[i + 1]) + ")" for i in range(len(bins) - 1)]
    ind = np.arange(len(counts[0]))
    for i in range(len(counts)):
        ax1.bar(ind + width2, counts[i], width=width, color = color_index[i],
                bottom=np.sum(counts[:i], axis=0), tick_label=bins_name, align='center', label=label_name_2[i])
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
    plt.title(titlen)
    plt.savefig(rst_path + '\\bp3')
    print "BP2: ",
    for each in counts_sum_2:
        print "{0:7.0f} | ".format(each),
    print " total: {0}".format(np.sum(counts_sum_2))
    print "     ",
    for each in counts_sum_2:
        print "{0:6.2f}% | ".format(each/np.sum(counts_sum_2)*100),
    print " total: 100%"
    print "大于{0:.0f}天的sku占{1:.2f}%。".format(binsn[3], np.sum(counts_sum_2[3:])/np.sum(counts_sum_2)*100)
    # table
    bp2_output = [[u'BP值范围',u'小于10天',u'10-20天',u'20-30天',u'30-40天',u'40-50天',u'50-60天',u'大于60天',u'总数']]
    bp2_output.append([u'数量']+map(lambda x: unicode('{0:.0f}'.format(x)),np.append(counts_sum_2,np.sum(counts_sum_2))))
    bp2_output.append([u'比例']+map(lambda x: unicode('{0:.2f}%'.format(x/np.sum(counts_sum_2)*100)),np.append(counts_sum_2,np.sum(counts_sum_2))))
    bp2_output2 = u"大于{0:.0f}天的sku占{1:.2f}%。".format(binsn[3], np.sum(counts_sum_2[3:])/np.sum(counts_sum_2)*100)
    return z_output, z_output2, bp1_output, bp1_output2, bp2_output, bp2_output2


def plotQuantile(df, kpi, filePath):
    # df = data
    # kpi = kpi_frame
    rst_path = get_path(filePath)
    '''
    将sku分为高周转率高周现货率，高周转率低现货率，低周转率高现货率，低周转率低现货率四个部分进行分析，画出库存与销量的变化曲线
    '''
    kpi = kpi[(kpi.TD.notnull())&(kpi.TD != np.inf)&(kpi.TD != -np.inf)&(kpi.CR.notnull())&(kpi.CR != np.inf)&(kpi.CR != -np.inf)]
    high_td = kpi.TD.quantile(0.75)
    low_td = kpi.TD.quantile(0.25)
    high_cr = kpi.CR.quantile(0.75)
    low_cr = kpi.CR.quantile(0.25)

    if os.path.exists(rst_path + '\\high_cr_high_td') == False:
        os.mkdir(rst_path + '\\high_cr_high_td')
    # print kpi[kpi.TD >= high_td][kpi.CR >= high_cr].index
    idlist = kpi[(kpi.TD >= high_td) & (kpi.CR >= high_cr)].index
    num = 0
    for id in idlist:
        # id = idlist[0]
        if num > 10:
            break
        sku = df[df.item_sku_id == id]
        plotsku(sku, kpi, rst_path + '\\high_cr_high_td')
        num += 1

    if os.path.exists(rst_path + '\\low_cr_high_td') == False:
        os.mkdir(rst_path + '\\low_cr_high_td')
    idlist = kpi[(kpi.TD >= high_td) & (kpi.CR <= low_cr)].index
    num = 0
    for id in idlist:
        if num > 60:
            break
        sku = df[df.item_sku_id == id]
        plotsku(sku, kpi, rst_path + '\\low_cr_high_td')
        num += 1

    if os.path.exists(rst_path + '\\high_cr_low_td') == False:
        os.mkdir(rst_path + '\\high_cr_low_td')
    idlist = kpi[(kpi.TD <= low_td) & (kpi.CR >= high_cr)].index
    num = 0
    for id in idlist:
        if num > 10:
            break
        sku = df[df.item_sku_id == id]
        plotsku(sku, kpi, rst_path + '\\high_cr_low_td')
        num += 1

    if os.path.exists(rst_path + '\\low_cr_low_td') == False:
        os.mkdir(rst_path + '\\low_cr_low_td')
    idlist = kpi[(kpi.TD <= low_td) & (kpi.CR <= low_cr)].index
    num = 0
    for id in idlist:
        if num > 10:
            break
        sku = df[df.item_sku_id == id]
        plotsku(sku, kpi, rst_path + '\\low_cr_low_td')
        num += 1


def plot_supp(z_value_frame, filePath):
    rst_path = get_path(filePath)

    fig1, ax1 = plt.subplots()
    # fig1 = plt.figure()
    ret =  plt.hist(z_value_frame.vlt, label='Z',bins=30, color='#0070C0')
    counts, bins, patches = ret[0], ret[1], ret[2]
    ax2 = ax1.twinx()
    sum_counts = np.cumsum(counts) / counts.sum()
    plt.plot(bins[1:], sum_counts, color='#C44E52')
    ax1.set_xlabel(u'供应商送货时长（天）')
    ax1.set_ylabel(u'频数')
    ax2.set_ylabel(u'供应商送货时长累积分布')
    ax1.set_ylim(-counts.max() * 0.05, counts.max() * 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.grid(False)
    plt.savefig(rst_path + '\\vlt')

    fig2, ax1 = plt.subplots()
    # fig2 = plt.figure()
    ret = plt.hist(z_value_frame.actual_plan_rate, label='Z', range=[0, 1], bins=30, color='#0070C0')
    counts, bins, patches = ret[0], ret[1], ret[2]
    ax2 = ax1.twinx()
    sum_counts = np.cumsum(counts) / counts.sum()
    plt.plot(bins[1:], sum_counts, '--', color='#C44E52', )
    ax1.set_xlabel(u'实际相对回告满足率')
    ax1.set_ylabel(u'频数')
    ax2.set_ylabel(u'实际相对回告满足率累积分布')
    ax1.set_ylim(-counts.max() * 0.05, counts.max() * 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.grid(False)
    # plt.xlim(0, 1)
    plt.savefig(rst_path + '\\actual_plan_rate')

    fig2, ax1 = plt.subplots()
    # fig3 = plt.figure()
    ret = plt.hist(z_value_frame.actual_origin_rate, label='Z', range=[0, 1], bins=30, color='#0070C0')
    counts, bins, patches = ret[0], ret[1], ret[2]
    ax2 = ax1.twinx()
    sum_counts = np.cumsum(counts) / counts.sum()
    plt.plot(bins[1:], sum_counts, '--', color='#C44E52')
    ax1.set_xlabel(u'实际满足率')
    ax1.set_ylabel(u'频数')
    ax2.set_ylabel(u'实际满足率累积分布')
    ax1.set_ylim(-counts.max() * 0.05, counts.max() * 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.grid(False)
    plt.savefig(rst_path + '\\actual_origin_rate')


def plothistper(plot_data, binsn, xlabeln, ylabeln, titlen, save_path, cum_True=True, size=(12,8), intshu=True):
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


def plotname(x):
    '''
    画zcase的对应情况图
    :param x: numeric z值
    :return: string 对应的名称
    '''
    if (x < 0):
        y = u"Z={0:.2f},补货过早".format(x)
    if (x >= 0) and (x <= 14):
        y = u"Z={0:.2f},补货适当".format(x)
    if (x > 14):
        y = u"Z={0:.2f},补货过晚".format(x)
    return y


def plotzcase(data, z_value_frame, filePath, N=20):
    '''
    画zcase图形
    :param data: pd.DataFrame 筛选后的数据
    :param z_value_frame: pd.DataFrame z值计算指标
    :param filePath: string 文件路径
    :param N: int 指定画zcase图的数量
    :return: None 仅用于作图
    '''
    rst_path = get_path(filePath)
    test = data.loc[:, ["item_sku_id", "day_string", "total_sales", "stock_qtty"]]
    test_grouped = test.groupby("item_sku_id")
    z_value_frame_need = z_value_frame["item_sku_id"].value_counts().index[:N]
    z_value_frame_temp = z_value_frame[map(lambda x: x in z_value_frame_need, z_value_frame["item_sku_id"])]
    test2 = z_value_frame_temp.loc[:, ["item_sku_id", "day_string", "z_value"]]
    test2.index = range(len(test2))
    test2["ZtextV"] = map(plotname, test2["z_value"].values)
    if os.path.exists(rst_path + '\\zcase') == False:
        os.mkdir(rst_path + '\\zcase')
    rst_path += '\\zcase'
    for item_sku_id in z_value_frame_need:
        # item_sku_id = z_value_frame_need[1]
        group = test_grouped.get_group(item_sku_id)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(group["day_string"], group["total_sales"], color="#6AB27B", label=u"总销量")
        ax.set_xlabel(u"时间")
        ax.set_ylabel(u"总销量")
        ymin1, ymax1 = plt.ylim()
        ax2 = ax.twinx()
        ax2.plot(group["day_string"], group["stock_qtty"], color="#0070C0", label=u"现货库存")
        ax2.set_ylabel(u"现货库存")
        ax2.yaxis.grid(False)
        ymin, ymax = plt.ylim()
        fig.autofmt_xdate()
        this_data = test2[test2["item_sku_id"] == item_sku_id]
        this_data.index = range(len(this_data))
        start = int(np.floor(len(this_data) / 3)) - 1
        step = range(start, len(this_data), start)[:-1]
        ymax *= 1.2
        annota_length = ymax / 6
        for i in step:
            # i = step[0]
            this_x = this_data["day_string"].iloc[i]
            this_y = group.loc[group["day_string"] == this_x, ["stock_qtty"]].iloc[0].values[0]
            plt.annotate(this_data["ZtextV"].iloc[i],
                         xy=(this_x, this_y),
                         xytext=(this_x, this_y + annota_length),
                         arrowprops=dict(width=1, facecolor='red', shrink=0.1, headwidth=7, headlength=10, color="red"),
                         color="r")
        plt.ylim((ymin, ymax))
        ax.set_ylim(ymin1 * 0.05, ymax1 * 1.05)
        ax.legend(loc=(.02, .94), fontsize=12, shadow=True)
        ax2.legend(loc=(.02, .9), fontsize=12, shadow=True)
        plt.title("Sku: {0}".format(item_sku_id))
        plt.savefig(rst_path + '\\{0}'.format(item_sku_id))


def plotcaigou(data, filePath, cum_True=True):
    '''
    画采购次数图
    :param data: pd.DataFrame 筛选后的数据
    :param filePath: string 文件路径
    :param cum_True: boolean 是否添加累计概率线
    :return: None 仅用于作图
    '''
    # filePath = analysis_path
    caigou_data = data[data.pur_bill_id.isnull() == False]["item_sku_id"].value_counts()
    binsn = range(19) + [np.inf]
    rst_path = get_path(filePath)
    save_path = rst_path + "\\caigou"
    xlabeln = u"采购次数"
    ylabeln = u"个数"
    titlen = u"采购次数分布"
    plothistper(caigou_data, binsn, xlabeln, ylabeln, titlen, save_path, cum_True=True)


def plotsupp2(supp_value_frame, analysis_path):
    '''
    用于输出csv文件和图
    :param supp_value_frame: pd.DataFrame 供应商方面数据
    :param analysis_path: string 文件路径
    :return: None 仅用于作图
    '''
    # 输出第一张表
    grouped = supp_value_frame.groupby('supp_name')
    summary = []
    for supp_name, group in grouped:
        caigousum = group["pur_bill_id"].drop_duplicates().count()
        skusum = group["item_sku_id"].drop_duplicates().count()
        vltmean = np.mean(group["vlt"])
        vltstd = np.std(group["vlt"])
        vltcv = vltstd / vltmean
        manzu = np.sum(group["actual_pur_qtty"]) / np.sum(group["originalnum"])
        manzu2 = np.nansum(group["actual_pur_qtty"]) / np.nansum(group["plan_pur_qtty"])
        summary.append({'supp_name':supp_name, 'buy_sum': caigousum,
                        'sku_sum': skusum, 'vlt_mean': vltmean, 'vlt_std': vltstd,
                        'vlt_cv': vltcv, 'full_rate':manzu, 'manzu2': manzu2})
    table1 = pd.DataFrame.from_dict(summary)
    calname = ['buy_sum','sku_sum','vlt_mean','vlt_std','vlt_cv','full_rate']
    for i, colname in enumerate(calname):
        min_num = np.min(table1[colname])
        max_num = np.max(table1[colname])
        cal_val = np.array(map(lambda x: (x - min_num)/(max_num - min_num)*9 + 1,table1[colname].values))
        if i == 0:
            sum_val = cal_val
        else:
            sum_val += cal_val
    sum_val /= 6
    table2 = pd.concat([table1, pd.DataFrame(sum_val.T, columns=["score_rank"])], axis=1)
    table2 = table2.loc[:,['supp_name','buy_sum','sku_sum','vlt_mean','vlt_std','vlt_cv','full_rate', 'score_rank']]
    table2 = table2.sort_values(['score_rank'], ascending=False)
    table2.to_csv(analysis_path + os.sep + 'supp_info.csv')

    # 画vlt图
    binsn = np.append(np.arange(0, 0.6, 0.05), 1)
    plothistper(table2['vlt_cv'], binsn, u'vltcv', u'个数', u'vltcv分布', analysis_path + '//report//vltcv', intshu=False)

    # 画满足率
    binsn2 = np.arange(0.3, 1.1, 0.1)
    plothistper(table1['full_rate'], binsn2, u'满足率', u'个数', u'实际满足率分布', analysis_path + '//report//manzu', intshu=False)
    plothistper(table1[table1['manzu2'].notnull()]["manzu2"], binsn2, u'满足率', u'个数', u'实际相对回告满足率分布', analysis_path + '//report//manzu2', intshu=False)
    manzu_output1 = u"\t实际满足率的基本统计信息，最小值:{0:.2f}%,中位数:{1:.2f}%,均值:{2:.2f}%,最大值:{3:.2f}%。".format(
        np.nanmin(table1[table1['full_rate']>0.01]['full_rate'].values)*100,np.nanmedian(table1['full_rate'].values)*100,
        np.nanmean(table1['full_rate'].values)*100,np.nanmax(table1['full_rate'].values)*100)
    print manzu_output1
    manzu_output2 = u"\t实际相对回告满足率分布的基本统计信息，最小值:{0:.2f}%,中位数:{1:.2f}%,均值:{2:.2f}%,最大值:{3:.2f}%。".format(
        np.nanmin(table1[table1['manzu2']>0.01]['manzu2'].values)*100,np.nanmedian(table1['manzu2'].values)*100,
        np.nanmean(table1['manzu2'].values)*100,np.nanmax(table1['manzu2'].values)*100)
    print manzu_output2
    # 画 vltcv * 满足率 图
    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.grid()
    ax.set_xlabel(u"vlt波动性")
    ax.set_ylabel(u"平均满足率")
    ax.set_xlim(0.05, 0.5)
    ax.set_ylim(0.2, 1.1)
    plt.scatter(table1['vlt_cv'], table1['full_rate'], color='#0070C0')
    plt.plot([0.3, 0.3], [0.2, 1.1], '--', color='red')
    plt.plot([0.05, 0.5], [0.8, 0.8], '--', color='red')
    plt.annotate('(1)', xy=(0.16, 0.9), fontsize=20, color='red')
    plt.annotate('(2)', xy=(0.39, 0.9), fontsize=20, color='red')
    plt.annotate('(3)', xy=(0.16, 0.5), fontsize=20, color='red')
    plt.annotate('(4)', xy=(0.39, 0.5), fontsize=20, color='red')
    plt.title(u'vlt稳定性与订单满足率散点图')
    plt.savefig(analysis_path + '//report//vltcv_manzu')
    return table2,manzu_output1,manzu_output2


def calcallback(func, sim_data, his_name, sim_name):
    '''
    获取不同band的数据
    :param func: 回调函数 用于计算的指标，如：np.nanmean
    :param sim_data: pd.DataFrame 仿真的数据sim_data
    :param his_name: string 历史字段名
    :param sim_name: string 仿真字段名
    :return: 表格的list
    '''
    his_zon = func(sim_data[his_name])
    pbs_zon = func(sim_data[sim_name])
    his_ad = func(sim_data[map(lambda x: x in list('ABCD'), sim_data["org_nation_sale_num_band"].values)][his_name])
    sim_ad = func(sim_data[map(lambda x: x in list('ABCD'), sim_data["org_nation_sale_num_band"].values)][sim_name])
    his_e = func(sim_data[map(lambda x: x in list('E'), sim_data["org_nation_sale_num_band"].values)][his_name])
    sim_e = func(sim_data[map(lambda x: x in list('E'), sim_data["org_nation_sale_num_band"].values)][sim_name])
    his_l = func(sim_data[map(lambda x: x in list('L'), sim_data["org_nation_sale_num_band"].values)][his_name])
    sim_l = func(sim_data[map(lambda x: x in list('L'), sim_data["org_nation_sale_num_band"].values)][sim_name])
    his_z = func(sim_data[map(lambda x: x in list('Z'), sim_data["org_nation_sale_num_band"].values)][his_name])
    sim_z = func(sim_data[map(lambda x: x in list('Z'), sim_data["org_nation_sale_num_band"].values)][sim_name])
    row_data = [his_zon, pbs_zon, his_ad, sim_ad, his_e, sim_e, his_l, sim_l, his_z, sim_z]
    return row_data


def getdata(sim_data, name_str):
    '''
    获得name_str的band数据
    :param sim_data: pd.DataFrame 仿真的数据sim_data
    :param name_str: string band的名称
    :return:
    '''
    return sim_data[map(lambda x: x in list(name_str), sim_data["org_nation_sale_num_band"].values)]


def plotsim(sim_path, analysis_path, sim_num_zon):
    '''
    画仿真的图
    :param sim_path: string 仿真report的路径
    :param analysis_path: string 分析路径
    :param sim_num_zon: numeric 用于仿真的sku数量
    :return: None 仅用于作图
    '''
    sim_data_path = sim_path + os.sep + 'sample_data_base_policy_kpi.csv'
    sim_data = pd.read_table(sim_data_path)
    sim_data2 = sim_data[sim_data["ito_sim"] != np.inf]
    sim_data2.index = range(len(sim_data2))
    # 分band
    row_1 = ['total', 'total', 'A-D', 'A-D','E','E','L','L','Z','Z']
    # 现货率
    row_2 = calcallback(np.nanmean, sim_data2, "cr_his", "cr_sim")
    # 周转 - 中位数
    row_3 = calcallback(np.nanmedian, sim_data2, "ito_his", "ito_sim")
    # 周转 - 均值
    row_4 = calcallback(np.nanmean, sim_data2, "ito_his", "ito_sim")
    # 金额维度
    row_5 = calcallback(np.sum, sim_data2, "gmv_his", "gmv_sim")
    table_data = [row_1,row_2,row_3,row_4,row_5]
    table1 = pd.DataFrame(table_data, columns=["his_total","pbs_total","his_ad","sim_ad","his_e","sim_e",
                                               "his_l","sim_l","his_z","sim_z"])
    table1.to_csv(analysis_path + os.sep + 'pbs_table1.csv')
    table3 = table1.T.reset_index()
    str_tmp = '应用PBS策略，两者总体情况对比发现：现货率和金额维度分别提升{0}%和2.34%。周转中位数相比实际情况下降19.98%，均值下降10.53%。'
    # 1、画周转变化图
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel(u"历史周转")
    ax1.set_ylabel(u"仿真周转")
    ax1.set_xlim(-10,410)
    ax1.set_ylim(-10,410)
    name_strs = ['ABCD','E','L','Z']
    ## 亮色：橙色#F8766D，绿色#7CAE00，青色#00BFC4，紫色#C77CFF。
    color_map = ['#F8766D','#7CAE00','#00BFC4','#C77CFF']
    for i, name_str in enumerate(name_strs):
        ax1.scatter(getdata(sim_data, name_str)['ito_his'], getdata(sim_data, name_str)['ito_sim'], color=color_map[i], label=name_str)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5), title='band')
    ax1.plot([0, 400], [0, 400], color='red')
    # sim_num_zon = 958
    sim_ito_num = Counter(sim_data['ito_his']>sim_data['ito_sim'])[True]
    his_ito_num = sim_num_zon - sim_ito_num
    plt.annotate(u'sku:{0:.0f}个'.format(his_ito_num), xy=(100, 300), fontsize=15, color='red')
    plt.annotate(u'sku:{0:.0f}个'.format(sim_ito_num), xy=(300, 100), fontsize=15, color='red')
    plt.title(u'周转变化情况')
    plt.savefig(analysis_path + '//report//ito_his_sim')
    # 2、画现货率图
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel(u"历史现货率")
    ax1.set_ylabel(u"仿真现货率")
    ax1.set_xlim(-0.05,1.05)
    ax1.set_ylim(-0.05,1.05)
    for i, name_str in enumerate(name_strs):
        ax1.scatter(getdata(sim_data, name_str)['cr_his'], getdata(sim_data, name_str)['cr_sim'], color=color_map[i], label=name_str)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5), title='band')
    ax1.plot([0, 1], [0, 1], color='red')
    sim_ito_num = Counter(sim_data['cr_his']>sim_data['cr_sim'])[True]
    his_ito_num = sim_num_zon - sim_ito_num
    plt.annotate(u'sku:{0:.0f}个'.format(his_ito_num), xy=(0.2, 0.7), fontsize=15, color='red')
    plt.annotate(u'sku:{0:.0f}个'.format(sim_ito_num), xy=(0.8, 0.3), fontsize=15, color='red')
    plt.title(u'现货率变化情况')
    plt.savefig(analysis_path + '//report//cr_his_sim')
    return table3


