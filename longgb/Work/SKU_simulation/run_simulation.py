# coding: utf-8
import os
from sys import path
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
import ast
import datetime
import logging
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from com.jd.test.zhangjs.SKU_simulation.SkuSimulation import SkuSimulation
from com.jd.test.zhangjs.SKU_simulation import configServer
import time
from StatisUtil import EMsmooth
import sys


# ============================== 功能函数 ============================
def fill_nan(input_array):
    '''
    向下填充和均值填充第一个
    '''
    if isinstance(input_array, np.ndarray):
        for idx in range(len(input_array)):
            if idx == 0 and np.isnan(input_array[idx]):
                input_array[idx] = np.nanmean(input_array)
            elif np.isnan(input_array[idx]):
                input_array[idx] = input_array[idx - 1]
    else:
        print 'Input must be 1d numpy.ndarray!'


# ======================================================================
# =                              0、配置信息                            =
# ======================================================================
# ----------------------------- 基本配置信息 ----------------------------
workingfolderName=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
data_dir0 = configServer.data_dir
output_dir = configServer.output_dir
data_file_name=configServer.data_file_name
logging_dir=configServer.log_dir
# ----------------------------- 本次仿真配置信息 ------------------------
write_daily_data = True
write_original_data = True
persistence = True
output_fig = True
simulation_name = 'sample_data_base_policy'
logging.basicConfig(filename=logging_dir + simulation_name + '.log', level=logging.INFO)


# ======================================================================
# =                              1、读入数据                            =
# ======================================================================
# 初始化数据
complete_sku = 0
train_kpi_list = []
simulation_results = {}
starttime = datetime.datetime.now()     # 今天日期
date_end=configServer.date_end
date_start=configServer.date_start
train_date_range = configServer.date_range[0]

# 1.0 读取数据
dtype = {'sku_id': str}
df = pd.read_csv(data_dir0 + data_file_name, sep='\t', dtype=dtype)

# 1.1 将缺失值填充为0
df.fillna(0,inplace=True)

# 1.2 对销量(sales)进行平滑
df_smooth_pre = df.loc[:,["sku_id","dt","total_sale"]]
df_smooth_pre.columns = ["sku","date","sales"]
df_smooth = EMsmooth.smooth(df_smooth_pre)
df["dt"]=df["dt"].astype(str)
df_smooth["date"]=df_smooth["date"].astype(str)
df=pd.merge(df,df_smooth,how='inner',left_on=["sku_id","dt"],right_on=["sku","date"])
df=df.loc[:,["fdc_id","sku_id","ofdsales","stock_qtty","dt","sales_smooth"]]
df.columns=["fdc_id","sku_id","ofdsales","stock_qtty","dt","total_sale"]

# 1.3 获取SKU列表
sku_list = df.sku_id.unique()
total_sku = len(sku_list)   # sku总数量值

# 1.4 计算VLT分布，采用历史分布数据
vlt_val = configServer.vlt_val
vlt_prob = configServer.vlt_prob
print u'开始循环遍历SKU'
for sku_id in sku_list[16:19]:
    # 设置仿真开始日期和仿真结束日期
    # 准备数据
    df_sku = df[df.sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.dt <train_date_range[0]) | (df_sku.dt > train_date_range[1])].index)
    # 排除不进行仿真情况
    if len(df_sku.ofdsales)<=0:
        logging.info(str(sku_id) + ': ' + '仿真开始日期销量预测数据为空，不进行仿真！')
        continue
    if (not isinstance(df_sku.ofdsales.iloc[0], str)):
        logging.info(str(sku_id) + ': ' + '仿真开始日期销量预测数据为空，不进行仿真！')
        continue
    if np.isnan(df_sku.stock_qtty.iloc[0]):
        logging.info(str(sku_id) + ': ' + '仿真开始日期库存数据为空，不进行仿真！')
        continue
    if abs(np.sum(df_sku.stock_qtty) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + '仿真期间总库存为0，也不仿真！')
        continue
    if abs(np.sum(df_sku.total_sale) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + '仿真期间总销量为0，不进行仿真！')
        continue
    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + '_origin.csv', index=False, sep='\t')

    sku_name = ''
    print(sku_id + '@' + sku_name + u': 开始仿真,关键KPI:sku_id,cr_sim,cr_his,ito_sim,ito_his,ts_sim,ts_his,pur_cnt_sim,s,S,date_range,ito_level')
    logging.info(str(sku_id) + '@' + sku_name + u': 开始仿真......')

    sales_his = df_sku.total_sale.as_matrix()
    inv_his = df_sku.stock_qtty.as_matrix()
    sales_pred_mean = []
    for x in df_sku.ofdsales:
        if type(x) == float and math.isnan(x):
            sales_pred_mean.append(None)
        else:
            sales_pred_mean.append(ast.literal_eval(x))
    ############ 利用搜索算法搜索最优参数 #####################
    # 先不遍历时间窗口和S-s,上面涉及 时间窗口的数据判断，因此把时间循环放到SKU循环下面一层
    for ito_level in range(5,11):
        for dis_s_S in np.arange(2,6,0.5):
            s=10
            S=s+dis_s_S
            while s>2:
                # 1 初始化
                sku_simulation = SkuSimulation(train_date_range, sales_his, inv_his, sales_pred_mean, vlt_val,
                                                   vlt_prob,s=s,S=S, sku_id=sku_id, sku_name=sku_name,ito_level=ito_level)
                logging.info(str(sku_id) + '@' + 'run_simulation()')
                # 2 运行仿真
                sku_simulation.run_simulation(seed=66)
                # 3 计算相关KPI
                sku_kpi=sku_simulation.calc_kpi()
                sku_kpi.append(-99)
                # print sku_kpi
                #根据KPI判断是否保留参数
                if sku_kpi[1]>=sku_kpi[2]:
                    if sku_kpi[3]<=sku_kpi[4]:
                        ss=s
                        sku_kpi[-1]=ss
                        # print sku_kpi
                        s=s-1
                        train_kpi_list.append(sku_kpi)
                        continue
                    else:
                        s=s-1
                else:
                    s=s-1
                    # break
                train_kpi_list.append(sku_kpi)
                simulation_results[sku_id] = sku_simulation

                # 将仿真明细数据输出到文件
                if write_daily_data:
                    daily_data = sku_simulation.get_daily_data()
                    daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '.csv', index=False, sep='\t')

                    # # 将图片输出到文件
                    # if output_fig:
                    #     sku_simulation.get_report(output_dir)

                    # 运行日志
                endtime = datetime.datetime.now()
                used_seconds = (endtime - starttime).total_seconds()
                complete_sku += 1

                logging.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                                 str(used_seconds))
# SKU粒度的KPI
kpi_df = pd.DataFrame.from_records(train_kpi_list, columns=['sku_id','cr_sim','cr_his','ito_sim','ito_his','ts_sim',
                                                      'ts_his','pur_cnt_sim','s','S','date_begin','date_end',
                                                      'ito_level','ts_rate','target_s'])
kpi_df.to_csv(path_or_buf=output_dir + simulation_name + '_kpi.csv', index=False, sep='\t')


Selected_s_S=kpi_df[kpi_df['target_s']>-99].loc[:,['sku_id','s','S']]
Selected_s_S['type']='sim'
System_s_S=pd.DataFrame({'sku_id':['997186','997160'],
                         's':[4,5],'S':[10,12]})
System_s_S['type']='system'
all_s_S=pd.concat([Selected_s_S,System_s_S])
##################################################################################################################################
###########################根据仿真训练结果在测试集上进行效果测试#################################################################
##################################################################################################################################
test_date_range=configServer.date_range[1]
test_sku_kpi=[]
test_simulation_results={}
for sku_id in sku_list[16:19]:
    # 设置仿真开始日期和仿真结束日期
    # 准备数据
    df_sku = df[df.sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.dt <test_date_range[0]) | (df_sku.dt > test_date_range[1])].index)

    # 如果仿真开始日期的销量预测数据为空，不进行仿真
    if len(df_sku.ofdsales)<=0:
        logging.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
        continue
    if (not isinstance(df_sku.ofdsales.iloc[0], str)):
        logging.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
        continue

    # 如果仿真开始日期的库存数据为空，不进行仿真
    if np.isnan(df_sku.stock_qtty.iloc[0]):
        logging.info(str(sku_id) + ': ' + u'仿真开始日期库存数据为空，不进行仿真！')
        continue

    #仿真期间总库存为0，也不仿真
    if abs(np.sum(df_sku.stock_qtty) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + u'仿真期间总库存为0，也不仿真！')
        continue

    # 如果仿真期间总销量为0，不进行仿真
    if abs(np.sum(df_sku.total_sale) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + u'仿真期间总销量为0，不进行仿真！')
        continue

    # sku_name = (df_sku.sku_name.iloc[0]).decode('gbk').encode('utf-8')
    sku_name = ''
    print(sku_id + '@' + sku_name + u': 开始仿真,关键KPI:sku_id,cr_sim,cr_his,ito_sim,ito_his,ts_sim,ts_his,pur_cnt_sim,s,S.date_range,ito_level')
    logging.info(str(sku_id) + '@' + sku_name + u': 开始仿真......')

    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + u'_origin.csv', index=False, sep='\t')

    sales_his = df_sku.total_sale.as_matrix()
    inv_his = df_sku.stock_qtty.as_matrix()

    sales_pred_mean = []
    for x in df_sku.ofdsales:
        if type(x) == float and math.isnan(x):
            sales_pred_mean.append(None)
        else:
            sales_pred_mean.append(ast.literal_eval(x))
            # print u'预测销量相关数据'
            # print sales_pred_mean
    Sku_all_s_S=all_s_S[all_s_S['sku_id']==sku_id]
    # print Sku_Selected_s_S
    for index,row in Sku_all_s_S.iterrows():
        s=row['s']
        S=row['S']
        sku_simulation = SkuSimulation(test_date_range, sales_his, inv_his, sales_pred_mean, vlt_val,
                                           vlt_prob,s=s,S=S, sku_id=sku_id, sku_name=sku_name)
        sku_simulation.run_simulation(seed=66)
        #计算相关KPI
        sku_kpi=sku_simulation.calc_kpi()
        sku_kpi.append(row['type'])
        print sku_kpi
        test_sku_kpi.append(sku_kpi)
        test_simulation_results[sku_id] = sku_simulation

        # 将仿真明细数据输出到文件
        if write_daily_data:
            daily_data = sku_simulation.get_daily_data()
            daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '.csv', index=False, sep='\t')

            # # 将图片输出到文件
            # if output_fig:
            #     sku_simulation.get_report(output_dir)

            # 运行日志
        endtime = datetime.datetime.now()
        used_seconds = (endtime - starttime).total_seconds()
        complete_sku += 1

        logging.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                     str(used_seconds))
# SKU粒度的KPI
kpi_df = pd.DataFrame.from_records(test_sku_kpi, columns=['sku_id','cr_sim','cr_his','ito_sim','ito_his','ts_sim',
                                                            'ts_his','pur_cnt_sim','s','S','date_begin','date_end',
                                                            'ito_level','ts_rate','target_s'])
kpi_df.to_csv(path_or_buf=output_dir + simulation_name + '_kpi.csv', index=False, sep='\t')
