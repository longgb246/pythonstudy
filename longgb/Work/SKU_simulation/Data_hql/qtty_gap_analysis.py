#-*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
import pandas as pd

read_path = r'D:\Lgb\data_sz'
save_path = r'D:\Lgb\WorkFiles\SKU_Allocations'

data1 = pd.read_table(read_path + os.sep + 'qtty_gap_analysis_01.out', header=None)
data2 = pd.read_table(read_path + os.sep + 'qtty_gap_analysis_02.out', header=None)
data3 = pd.read_table(read_path + os.sep + 'qtty_gap_analysis_03.out', header=None)

data1.columns = ['level', 'sku_num', 'diff']
data2.columns = ['level', 'sku_num', 'diff']
data3.columns = ['level', 'sku_num', 'diff']

data1.to_csv(save_path + os.sep + 'qtty_gap_analysis_01.csv', index=False)
data2.to_csv(save_path + os.sep + 'qtty_gap_analysis_02.csv', index=False)
data3.to_csv(save_path + os.sep + 'qtty_gap_analysis_03.csv', index=False)


plt.style.use('seaborn-darkgrid')
fig = plt.figure()
ax = fig.add_subplot(111)
y_pos = range(len(data1['sku_num']))
ax.barh(y_pos, data1['sku_num'])




