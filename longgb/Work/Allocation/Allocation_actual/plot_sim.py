#-*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.dates import  DateFormatter
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体


def plot2(data_tmp):
    data_tmp = a
    formatter = DateFormatter('%Y-%m-%d')
    ax = plt.figure(figsize=(18,12))
    ax1 = plt.subplot(211)  # 在图表2中创建子图1
    ax2 = plt.subplot(212)  # 在图表2中创建子图2
    data_tmp['date_s'] = pd.to_datetime(data_tmp['date_s'])
    ax1.plot(data_tmp['date_s'], data_tmp['x1'], label=u'实际库存')
    ax1.plot(data_tmp['date_s'], data_tmp['x2'], label=u'仿真库存')
    ax1.xaxis.set_major_formatter(formatter)
    ax1.legend()
    ax.savefig(r'F:\aa.png')



a1 = np.random.randn(30)
a1 = a1.cumsum()
a2 = np.random.randn(30)
a2 = a2.cumsum()
a3 = np.array(map(lambda x: str(x)[:10],pd.date_range('2016-10-02', '2016-10-31').values))

a = pd.DataFrame([a3,a1,a2]).T
a.columns = ['date_s', 'x1', 'x2']
plot2(a)

