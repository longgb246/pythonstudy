#-*- coding:utf-8 -*-
import os
import cPickle as pickle
import numpy as np
import pandas as pd
import time,datetime
import re
from matplotlib import pyplot as plt
from pylab import *
import  dateutil

if __name__=="__main__":
    #读入数据，读入gene_sample_data.py生成的抽样数据
    # sample_data=pd.read_csv("E:/Allocation_data/table_sample_sku.csv",date_parser='date_s')
    sample_data=pd.read_csv("E:/Allocation_data/table_sample_sku.csv",date_parser='date_s')
    sku_set=set(sample_data['sku_x'])
    fdc_set=set(sample_data['fdc'])
    matplotlib.rc('xtick',labelsize=8)
    matplotlib.rc('ytick',labelsize=10)
    for s in sku_set:
        for fdc in fdc_set:
            mask=(sample_data['sku_x']==s)&(sample_data['fdc']==fdc)
            tmp_data=sample_data[mask]
            tmp_data.set_index('date_s',inplace=True)
            tmp_data=tmp_data.sort_index()
            tmp_data_fdc_sale=tmp_data['sale_qtty_fdc'].rolling(window=7).sum()
            tmp_data_fdc_sale= pd.DataFrame(tmp_data_fdc_sale.shift(-7))
            tmp_data=pd.merge(tmp_data,tmp_data_fdc_sale,left_index=True,right_index=True)
            tmp_data=tmp_data.loc[:,['sku_x','inv','sale_qtty_fdc_x','sale_qtty_rdc_simu',
                                     'sale_qtty','mean_sales','sale_qtty_fdc_y']]
            tmp_data.columns=['sku','inv','sale_qtty_fdc','sale_qtty_simu',
                              'sale_qtty','sfs_mean_sales','real_mean_sales']
            plt.figure(1)
            p1 = plt.subplot(211)
            p2 = plt.subplot(212)
            # plt.plot([0, 1, 2, 3], [0.1, 0.2, 0.15, 0.3])
            tmp_data=tmp_data.reset_index()
            x_lab=list(d.strftime('%m-%d') for d in map(dateutil.parser.parse,tmp_data['date_s']))
            date_range_x=map(dateutil.parser.parse,tmp_data['date_s'])
            p1.set_xticks(date_range_x)
            # plt.plot(tmp_data['date_s'].astype(str),tmp_data['sale_qtty_fdc'])
            p1.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['sale_qtty_fdc'],linewidth=2)
            p1.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['inv'],linewidth=2)
            # p1.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['sale_qtty_fdc'],linewidth=2)
            # p1.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['inv'],linewidth=2)
            p1.set_xlabel(u"Date")
            p1.set_ylabel(u"Quantity")
            p1.set_title(u"Inv .. Sale")
            p1.legend()
            p1.title.set_fontsize(8)
            p1.set_xticklabels(x_lab, rotation=45, size=8)
            p1.set_ylim(0,np.max(tmp_data['inv'])+2)
            #绘画预测7天销量和实际销量对比图
            print x_lab
            p2.set_xticks(date_range_x)
            p2.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['sfs_mean_sales'],linewidth=2)
            p2.plot(map(dateutil.parser.parse,tmp_data['date_s']),tmp_data['real_mean_sales'],linewidth=2)
            # p2.plot(x_lab,tmp_data['mean_sales'],linewidth=2)
            # p2.plot(x_lab,tmp_data['real_mean_sales'],linewidth=2)
            p2.set_xlabel(u"Date")
            p2.set_ylabel(u"Quantity")
            p2.set_title(u"SFS_Sum_sales .. Real_Sum_sales")
            p2.legend()
            p2.title.set_fontsize(8)
            p2.set_xticklabels(x_lab, rotation=45, size=8)
            p2.set_ylim(0,np.max(tmp_data['inv'])+2)
            p2.set_xlim(np.min(date_range_x),np.max(date_range_x))
            # p2.set_xlim(np.min(tmp_data['date_s']),np.max(tmp_data['date_s']))
            save_path='E:/Allocation_data/test/'+str(fdc)+'_'+str(s)+'.png'
            plt.savefig(save_path)
            plt.close()