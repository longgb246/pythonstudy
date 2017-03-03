#-*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.dates import  DateFormatter
from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
import warnings
warnings.filterwarnings('ignore')


def plot2(data_tmp, save_path):
    # data_tmp = data_tmp_2
    formatter = DateFormatter('%Y-%m-%d')
    data_tmp = data_tmp.sort_values(['date_s'])
    ax = plt.figure(figsize=(18,12))
    ax1 = plt.subplot(211)  # 在图表2中创建子图1
    ax2 = plt.subplot(212)  # 在图表2中创建子图2
    data_tmp['date_s'] = pd.to_datetime(data_tmp['date_s'])
    ito_actual = np.sum(data_tmp['inv_fdc'])/(np.sum(data_tmp['sale_qtty_actual'])+0.00001)
    ito_sim = np.sum(data_tmp['inv'])/(np.sum(data_tmp['sale_qtty_sim'])+0.00001)
    ax1.plot(data_tmp['date_s'], data_tmp['inv_fdc'], label=u'实际库存')
    ax1.plot(data_tmp['date_s'], data_tmp['inv'], label=u'仿真库存', alpha=0.5)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_title(u'实际库存&仿真库存  ito_actual:{0:.2f} & ito_sim:{1:.2f}'.format(ito_actual, ito_sim))
    ax1.legend()
    ax2.plot(data_tmp['date_s'], data_tmp['sale_qtty_actual'], label=u'实际销售')
    ax2.plot(data_tmp['date_s'], data_tmp['sale_qtty_sim'], label=u'仿真销售', alpha=0.5)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_title(u'实际销售&仿真销售  ito_actual:{0:.2f} & ito_sim:{1:.2f}'.format(ito_actual, ito_sim))
    ax2.legend()
    ax.savefig(save_path)
    # canvas = FigureCanvas(ax)
    # canvas.print_figure('demo.jpg')


def mkdir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)


# a1 = np.random.randn(30)
# a1 = a1.cumsum()
# a2 = np.random.randn(30)
# a2 = a2.cumsum()
# a3 = np.array(map(lambda x: str(x)[:10],pd.date_range('2016-10-02', '2016-10-31').values))
#
# a = pd.DataFrame([a3,a1,a2]).T
# a.columns = ['date_s', 'x1', 'x2']
# plot2(a)

read_path = r'D:\Lgb\data_local'
save_path = r'D:\Lgb\data_local\sku_select_20170118'
if os.path.exists(save_path) == False:
    os.mkdir(save_path)


if __name__ == '__main__':
    # upper = [ 1280273,  928128, 2736969, 3303104,  818098,  189982, 2967897, 2956045, 2857483, 3093094, 2131674,  526835, 1150552, 1043580, 2141606, 2247316,  923635, 2655623, 1084057, 3343745]
    # lower = [ 852297,  884573,  246463, 1947864,  689229, 1039901, 2011989, 2202891, 3029042, 1450592,  688044,  688044,  312601,  385631, 3508620, 1085712,  766246, 3749844,  385665, 2069538]
    upper = pd.read_csv(read_path + os.sep + '628_per4_upper.csv')
    lower = pd.read_csv(read_path + os.sep + '628_per4_lower.csv')
    upper = list(set(map(str, upper['sku'].values) + ['1577266', '3067997']))
    lower = list(set(map(str, lower['sku'].values) + ['1577266', '3067997']))
    # print upper
    # print lower

    inv_path = read_path + os.sep + 'select_inv.out'
    sale_path = read_path + os.sep + 'select_sale.out'
    # 注意有改动
    # inv_data = pd.read_csv(inv_path)
    # sale_data = pd.read_csv(sale_path)
    inv_data = pd.read_table(inv_path, header=None)
    inv_data.columns = ['sku_id','open_po_fdc','inv_fdc','date_s','dc_id','date_s_y','fdc','sku','inv']
    sale_data = pd.read_table(sale_path, header=None)
    sale_data.columns = ['date_s','dc_id','item_sku_id','sale_qtty_actual','date_s_y','fdc','sku','sale_qtty_sim']
    inv_data['sku_id'] = map(str, inv_data['sku_id'].values)
    inv_data['dc_id'] = map(str, inv_data['dc_id'].values)
    inv_data = inv_data.groupby(['sku_id','dc_id','date_s'])['inv_fdc','inv'].sum().reset_index()


    sale_data = sale_data.rename(columns={'item_sku_id':'sku_id'})
    sale_data['sku_id'] = map(str, sale_data['sku_id'].values)
    sale_data['dc_id'] = map(str, sale_data['dc_id'].values)
    data = inv_data.merge(sale_data, on=['sku_id', 'dc_id','date_s'], how='left')
    data = data.fillna(0)
    # data.columns
    for fdc in ['628', '630', '658']:
        print fdc, '....'
        save_path_fdc = save_path + os.sep + fdc
        save_path_fdc_high = save_path_fdc + os.sep + 'sim_higher_actual'
        save_path_fdc_lower = save_path_fdc + os.sep + 'sim_lower_actual'
        mkdir(save_path_fdc)
        mkdir(save_path_fdc_high)
        mkdir(save_path_fdc_lower)
        # fdc = '628'
        data_tmp = data[data['dc_id'] == fdc]
        sku_tmp = list(set(data_tmp['sku_id']))
        for sku in sku_tmp:
            # sku = sku_tmp[0]
            data_tmp_2 = data_tmp[data_tmp['sku_id'] == sku]
            save_path_tmp = save_path_fdc_high if sku in upper else save_path_fdc_lower
            plot2(data_tmp_2, save_path_tmp + os.sep + '{0}_{1}_inv_sale'.format(fdc, sku))

    data.to_csv(save_path + os.sep + 'select_inv_sale.csv',index=False)


# data2967897_inv = inv_data[(inv_data['sku_id'] == '2967897')&(inv_data['dc_id'] == '628')].sort_values(['date_s'])
# data2967897 = data[(data['sku_id'] == '2967897')&(data['dc_id'] == '628')]
# data2967897 = data2967897.sort_values(['date_s'])
# np.sum(data2967897['inv_fdc'])
# np.sum(data2967897['inv'])
# data2967897.loc[:,['sku_id','inv_fdc','inv','date_s']]
# data2967897_inv
# np.sum(data2967897_inv['inv_fdc'])
# np.sum(data2967897_inv['inv'])


tuihuo_path = r'D:\Lgb\data_sz\hive_tuihuo.out'
tuihuo_data = pd.read_table(tuihuo_path, header=None)
tuihuo_data = tuihuo_data[(tuihuo_data[45] == '2016-10-02')|(tuihuo_data[45] == '2016-10-03')]
tuihuo_data.to_csv(r'D:\Lgb\data_local\hive_tuihuo.csv', index=False)




# 可能问题：lop、lop_replacememt较高，原因是mean_sales高。
#

