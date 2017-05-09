#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


path = r'D:\Lgb\data_sz'

kpi = pd.read_table(path + os.sep + 'kpi_online.txt')
datas = pd.read_table(path + os.sep + 'testData_online.txt')


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kpi['Cr'], kpi['Ito'], '.', color='#4C72B0', label='Online')
ax.plot(kpi['simuCr'], kpi['simuIto'], '.', color='#C44E52', label='OnlineSimu')
upper = np.ceil(np.nanpercentile(kpi['Ito'], 90))
lower = np.floor(np.nanpercentile(kpi['Ito'], 10))
ax.set_xlabel('Cr')
ax.set_ylabel('Ito')
ax.set_title('Cr and Ito : ( Online vs. OnlineSimu)')
ax.set_ylim(lower, upper)
ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(right=0.8)

# 为什么仿真的现货率会偏低？
# 思路：
# 找几个 CR 为 1的 skuRdc，跟踪分析为什么为变低。


datas
kpi

'10-1093041'


datas[datas['rdcSkuid']=='10-1093041'].to_csv(path + os.sep + 'result.csv', index=False)





if __name__ == '__main__':
    pass
