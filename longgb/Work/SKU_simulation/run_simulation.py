# coding: utf-8
import os
import ast
import datetime
import logging
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import time
import csv
import sys
from sys import path
pth = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
from StatisUtil import EMsmooth
from SkuSimulation import SkuSimulation
# import configServer


# ============================ 功能函数 ==============================
def fill_nan(input_array):
    if isinstance(input_array, np.ndarray):
        for idx in range(len(input_array)):
            if idx == 0 and np.isnan(input_array[idx]):
                input_array[idx] = np.nanmean(input_array)
            elif np.isnan(input_array[idx]):
                input_array[idx] = input_array[idx - 1]
    else:
        print 'Input must be 1d numpy.ndarray!'


# ====================================================================
# =                             0、配置信息                           =
# ====================================================================
# ---------------------------- 系统配置信息 ---------------------------
workingfolderName = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
data_dir0 = 'D:/Lgb/data_sz/allocation_data/'
output_dir = 'D:/Lgb/data_sz/allocation_data/'
data_file_name = 'half_month_sale_datasets_20170308221210 (3).csv'
para_file_name = 'sample_sku_param_20170221134526.csv'
logging_dir = 'D:/Lgb/data_sz/allocation_data/log/'


# ---------------------------- 本次仿真配置信息 ------------------------
write_daily_data = True
write_original_data = True
persistence = True
output_fig = True
train_simulation_name = 'train_kpi'
'''0,s min S min
   1,s min S max
   2,s max S min
   3,s max S max'''
sim_s_S_type = 4
logging.basicConfig(filename=logging_dir + train_simulation_name + '.log', level=logging.INFO)


# ====================================================================
# =                             1、数据读取                           =
# ====================================================================
# 1.0 读取数据
read_file = r'D:\Lgb\data_sz\allocation_data\allocation_data.out'
data = pd.read_table(read_file, header=None)
data.columns = ["fdc_id", "sku_id", "ofdsales", "total_sale", "stock_qtty", "dt", "safestock", "maxstock", "lag_window_0",
                "std_sale", "avg_sale", "av_sale", "total_sale_sum", "total_sale_per"]
df = data.copy()
df["sku_id"] = df["sku_id"].astype(str)
dtype = {'sku_id': str}
# 1.1 将缺失值填充为0
df.fillna(0, inplace=True)
# 1.2 对销量进行平滑
flag_smooth = 0
if flag_smooth == 1:
    print u'开始数据平滑处理'
    df_smooth_pre = df.loc[:, ["sku_id", "dt", "total_sale"]]
    df_smooth_pre.columns = ["sku", "date", "sales"]
    print u'调用平滑函数'
    df_smooth = EMsmooth.smooth(df_smooth_pre)
    print u'平滑函数处理完成'
    df["dt"] = df["dt"].astype(str)
    df_smooth["date"] = df_smooth["date"].astype(str)
    df = pd.merge(df, df_smooth, how='inner', left_on=["sku_id", "dt"], right_on=["sku", "date"])
    df = df.loc[:,
         ["fdc_id", "sku_id", "ofdsales", "stock_qtty", "dt", "sales_smooth", "safestock", "maxstock", "total_sale_per",
          "av_sale"]]
    df.columns = ["fdc_id", "sku_id", "ofdsales", "stock_qtty", "dt", "total_sale", "safestock", "maxstock",
                  'ts_percent', 'cv_sale']
    df.to_csv(path_or_buf=output_dir + 'smooth_datasets.csv', index=False, sep='\t')
    print u'数据平滑处理完成'
else:
    df = pd.read_csv(output_dir + 'smooth_datasets.csv', sep='\t', dtype=dtype)
# 1.3 获取SKU列表
sku_list = df.sku_id.unique()
# 1.4 记录哪些SKU没有找到最优的(s,S)
no_better_para_sku_list = list(sku_list)
total_sku = len(sku_list)
complete_sku = 0
train_kpi_list = []
simulation_results = {}
starttime = datetime.datetime.now()
date_end = '2017-01-30'
date_start = '2016-12-01'
train_date_range = ['2016-12-02','2016-12-15']
# 1.5 计算VLT分布，采用历史分布数据
vlt_val = np.array([1,2])
vlt_prob = np.array([0.9944,0.0056])
# 1.6 计算SKU的时间长度
train_start_dt = datetime.datetime.strptime(train_date_range[0], '%Y-%m-%d')
train_end_dt = datetime.datetime.strptime(train_date_range[1], '%Y-%m-%d')
trian_length = (train_end_dt - train_start_dt).days + 1

# df.isnull()
# np.sum()
# df_test = df.loc[map(lambda x: False if np.sum(x) == 0 else True,df.isnull().values),:]

# ====================================================================
# =                             2、仿真开始                           =
# ====================================================================
first = 0
if first == 1:
    # 2.0 开始仿真
    print u'开始循环遍历SKU'
    print u'SKU原始数量为:',len(sku_list)
    # 2.1 按照 sku 进行仿真
    for sku_id in sku_list:
        # sku_id = '1000054'

        # df_sku = df[df.sku_id == sku_id]
        # df_sku = df_sku.drop(df_sku[(df_sku.dt < train_date_range[0]) | (df_sku.dt > train_date_range[1])].index)
        # if (df_sku.shape[0]>=trian_length) and (len(df_sku.ofdsales)>1) and (isinstance(df_sku.ofdsales.iloc[0], str)) and (not np.isnan(df_sku.stock_qtty.iloc[0])) and (abs(np.sum(df_sku.stock_qtty) - 0) > 1e-3) and (abs(np.sum(df_sku.total_sale) - 0) > 1e-3):
        #     print sku_id
        #     break

        # 2.2 提取 2016-12-02 ~ 2016-12-15 之间的数据
        df_sku = df[df.sku_id == sku_id]
        df_sku = df_sku.drop(df_sku[(df_sku.dt <train_date_range[0]) | (df_sku.dt > train_date_range[1])].index)
        # 2.3 判断情况不进行仿真：
        #       1、仿真日期不满14天的；
        #       2、ofdsales的数据长度小于等于1的；      3、ofdsales的第一个元素部位str的；
        #       4、stock_qtty的第一个元素为空的；       5、stock_qtty的求和（没有考虑正负项）为0的；
        #       6、total_sale的求和（没有考虑正负项）为0的；
        if df_sku.shape[0]<trian_length:
            continue
        if len(df_sku.ofdsales)<=1:
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
        sku_name = ''
        logging.info(str(sku_id) + '@' + sku_name + u': 开始仿真......')
        # 2.4 将原始数据输出到文件
        if write_original_data:
            df_sku.to_csv(path_or_buf=output_dir + 'origin/'+ str(sku_id) + '_origin.csv', index=False, sep='\t')
        sales_his = df_sku.total_sale.as_matrix()
        inv_his = df_sku.stock_qtty.as_matrix()
        sales_per=df_sku.ts_percent.as_matrix()
        cv_sale=df_sku.cv_sale.as_matrix()
        cv_sale=cv_sale[0]
        sales_pred_mean = []
        for x in df_sku.ofdsales:
            if type(x) == float and math.isnan(x):
                sales_pred_mean.append(None)
            else:
                sales_pred_mean.append(ast.literal_eval(x))
        ############ 2.5 利用搜索算法搜索最优参数 #####################
        # 先不遍历时间窗口和S-s,上面涉及 时间窗口的数据判断，因此把时间循环放到SKU循环下面一层
        for ito_level in range(5,6):
            # ito_level = 5
            for dis_s_S in np.arange(4,8,0.5):
                # dis_s_S = 4
                s=8
                while s>3:
                    S=s+dis_s_S
                    # 2.5.1 初始化
                    sku_simulation = SkuSimulation(train_date_range, sales_his, inv_his, sales_pred_mean, vlt_val,
                                                       vlt_prob,s=s,S=S, sku_id=sku_id, sku_name=sku_name,ito_level=ito_level,
                                                       sale_per=sales_per,cv_sale=cv_sale)
                    logging.info(str(sku_id) + '@' + 'run_simulation()')
                    sku_simulation.run_simulation(seed=66)
                    # 2.5.2 计算相关KPI
                    sku_kpi = sku_simulation.calc_kpi()
                    sku_kpi.append(-99)
                    # print sku_kpi
                    #根据KPI判断是否保留参数
                    if sku_kpi[1]>=math.sqrt(sku_kpi[2]):
                        if ((sku_kpi[4]>0 and float(sku_kpi[3]-sku_kpi[4])/float(sku_kpi[4])<=0.08) or sku_kpi[3]<=7):
                            ss=s
                            sku_kpi[-1]=ss
                            # print sku_kpi
                            s=s-1
                            train_kpi_list.append(sku_kpi)
                            if sku_id in no_better_para_sku_list:
                                no_better_para_sku_list.remove(sku_id)
                            continue
                        else:
                            s=s-1
                    else:
                        s=s-1
                    train_kpi_list.append(sku_kpi)          # 【这里为什么添加2次】
                    simulation_results[sku_id] = sku_simulation
                    # 将仿真明细数据输出到文件
                    if write_daily_data:
                        daily_data = sku_simulation.get_daily_data()
                        daily_data.to_csv(path_or_buf=output_dir + 'daily_data/' + str(sku_id) + '.csv', index=False, sep='\t')
                    endtime = datetime.datetime.now()
                    used_seconds = (endtime - starttime).total_seconds()
                    complete_sku += 1
                    logging.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                                     str(used_seconds))
    # SKU粒度的KPI
    kpi_df = pd.DataFrame.from_records(train_kpi_list, columns=['sku_id','cr_sim','cr_his','ito_sim','ito_his','ts_sim',
                                                          'ts_his','pur_cnt_sim','s','S','date_begin','date_end',
                                                          'ito_level','ts_rate','ts_percent','cv_sale','target_s'])
    kpi_df.to_csv(path_or_buf=output_dir + train_simulation_name + '_kpi.csv', index=False, sep='\t')
    ### 将没有找到最优解备选集合的SKU，进行保存便于分析原因
    with open(output_dir +'no_better_para_sku_list.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(no_better_para_sku_list)
    print 'Over !!!'
    exit()


# =====================================================================
# =                         4、这里提取数据                            =
# =====================================================================
kpi_df = pd.read_csv(output_dir + train_simulation_name + '_kpi.csv', sep='\t')

test_simulation_name = 'test_kpi_'
# 发生调拨行为了
mask = (kpi_df['target_s'] > (-99)) & (kpi_df['pur_cnt_sim'] > 0)
Selected_s_S = kpi_df[mask].loc[:,
               ['sku_id', 's', 'S', 'cr_sim', 'ito_sim', 'ito_his', 'ts_his', 'ts_percent', 'cv_sale']]
Selected_s_S['sku_id'] = Selected_s_S['sku_id'].astype(str)
# 需要在仿真中增加一列传输数据
# Selected_s_S['ts_percent']=1
'''0,s min S min
   1,s min S max
   2,s max S min
   3,s max S max
   4,mixture s,S'''
s_S_type = ''
if sim_s_S_type == 0:
    tmp1 = Selected_s_S.groupby(['sku_id']).min()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]
    tmp3 = pd.merge(tmp2, Selected_s_S, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    '''S的取值'''
    # tmp4=tmp3.groupby(['sku_id','s']).min()
    tmp4 = tmp3.groupby(['sku_id', 's']).min()
    Selected_s_S = tmp4.reset_index()
    s_S_type = 's_min_S_min'
elif sim_s_S_type == 1:
    tmp1 = Selected_s_S.groupby(['sku_id']).min()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]
    tmp3 = pd.merge(tmp2, Selected_s_S, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    '''S的取值'''
    # tmp4=tmp3.groupby(['sku_id','s']).min()
    tmp4 = tmp3.groupby(['sku_id', 's']).max()
    Selected_s_S = tmp4.reset_index()
    s_S_type = 's_min_S_max'
elif sim_s_S_type == 2:
    tmp1 = Selected_s_S.groupby(['sku_id']).max()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]
    tmp3 = pd.merge(tmp2, Selected_s_S, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    '''S的取值'''
    # tmp4=tmp3.groupby(['sku_id','s']).min()
    tmp4 = tmp3.groupby(['sku_id', 's']).min()
    Selected_s_S = tmp4.reset_index()
    s_S_type = 's_max_S_min'
elif sim_s_S_type == 3:
    tmp1 = Selected_s_S.groupby(['sku_id']).max()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]
    tmp3 = pd.merge(tmp2, Selected_s_S, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    '''S的取值'''
    # tmp4=tmp3.groupby(['sku_id','s']).min()
    tmp4 = tmp3.groupby(['sku_id', 's']).max()
    Selected_s_S = tmp4.reset_index()
    s_S_type = 's_max_S_max'
elif sim_s_S_type == 4:
    tmp_s_S_list = []
    # 针对高销量的sku的处理，s取最大，按照周转选择S，当周转小于5,采用最大的S，当周转大于5，则采用较小的S
    Selected_s_S_mid_01 = Selected_s_S[Selected_s_S['ts_percent'] <= 0.5]           # ts_percent小于等于0.5
    tmp1 = Selected_s_S_mid_01.groupby(['sku_id']).max()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]                             # s取最大。
    tmp3 = pd.merge(tmp2, Selected_s_S_mid_01, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    tmp4_more_5 = tmp3.groupby(['sku_id', 's']).max()
    tmp4_more_5.reset_index(inplace=True)
    tmp4_more_5 = tmp4_more_5[tmp4_more_5['ito_his'] <= 5]          # ito_his 最大小于5
    tmp_s_S_list.append(tmp4_more_5)
    tmp4_less_5 = tmp3.groupby(['sku_id', 's']).min()
    tmp4_less_5.reset_index(inplace=True)
    tmp4_less_5 = tmp4_less_5[tmp4_less_5['ito_his'] > 5]           # ito_his 最小大于5
    tmp_s_S_list.append(tmp4_less_5)

    # 针对低销量的SKU选择s最小的SKU，最小的S
    Selected_s_S_mid_02 = Selected_s_S[Selected_s_S['ts_percent'] >= 0.99]
    tmp1 = Selected_s_S_mid_02.groupby(['sku_id']).min()
    tmp1.reset_index(inplace=True)
    tmp2 = tmp1.loc[:, ['sku_id', 's']]                             # s取最小。
    tmp3 = pd.merge(tmp2, Selected_s_S_mid_02, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
    '''S的取值'''
    # tmp4=tmp3.groupby(['sku_id','s']).min()
    tmp4 = tmp3.groupby(['sku_id', 's']).min()
    Selected_s_S_mid_02 = tmp4.reset_index()
    tmp_s_S_list.append(Selected_s_S_mid_02)

    ##销量分位点位于50%-80%之间的SKU，选择次大的s,当周转小于8,采用最大的S，当周转大于8，则采用较小的S
    mask = (Selected_s_S['ts_percent'] > 0.5) & (Selected_s_S['ts_percent'] <= 0.8)
    Selected_s_S_mid_03 = Selected_s_S[mask]
    mask = Selected_s_S_mid_03.rank(method='min')['s'] > 1
    Selected_s_S_mid_03 = Selected_s_S_mid_03[mask]
    # 如果备选集合中不同s的数量大于1
    if Selected_s_S_mid_03.shape[0] > 0:
        tmp1 = Selected_s_S_mid_03.groupby(['sku_id']).max()
        tmp1.reset_index(inplace=True)
        tmp2 = tmp1.loc[:, ['sku_id', 's']]
        tmp3 = pd.merge(tmp2, Selected_s_S_mid_03, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
        tmp4_more_8 = tmp3.groupby(['sku_id', 's']).max()
        tmp4_more_8.reset_index(inplace=True)
        tmp4_more_8 = tmp4_more_8[tmp4_more_8['ito_his'] <= 8]
        tmp_s_S_list.append(tmp4_more_8)
        tmp4_less_8 = tmp3.groupby(['sku_id', 's']).min()
        tmp4_less_8.reset_index(inplace=True)
        tmp4_less_8 = tmp4_less_8[tmp4_less_8['ito_his'] > 8]
        tmp_s_S_list.append(tmp4_less_8)
    # 如果备选集合中不同s的数量等于1
    else:
        mask = (Selected_s_S['ts_percent'] > 0.5) & (Selected_s_S['ts_percent'] <= 0.8)
        tmp3 = Selected_s_S[mask]
        tmp4_more_8 = tmp3.groupby(['sku_id', 's']).max()
        tmp4_more_8.reset_index(inplace=True)
        tmp4_more_8 = tmp4_more_8[tmp4_more_8['ito_his'] <= 8]
        tmp_s_S_list.append(tmp4_more_8)
        tmp4_less_8 = tmp3.groupby(['sku_id', 's']).min()
        tmp4_less_8.reset_index(inplace=True)
        tmp4_less_8 = tmp4_less_8[tmp4_less_8['ito_his'] > 8]
        tmp_s_S_list.append(tmp4_less_8)

    # 销量分位点位于80%-98%之间的SKU，选择次小的s,当周转小于10,采用最大的S，当周转大于10，则采用较小的S
    mask = (Selected_s_S['ts_percent'] > 0.8) & (Selected_s_S['ts_percent'] <= 0.98)
    Selected_s_S_mid_04 = Selected_s_S[mask]
    mask = Selected_s_S_mid_04.rank(method='min')['s'] > 1
    Selected_s_S_mid_04 = Selected_s_S_mid_04[mask]
    # 如果备选集合中不同s的数量大于1
    if Selected_s_S_mid_04.shape[0] > 0:
        tmp1 = Selected_s_S_mid_04.groupby(['sku_id']).min()
        tmp1.reset_index(inplace=True)
        tmp2 = tmp1.loc[:, ['sku_id', 's']]
        tmp3 = pd.merge(tmp2, Selected_s_S_mid_04, left_on=['sku_id', 's'], right_on=['sku_id', 's'])
        tmp4_more_10 = tmp3.groupby(['sku_id', 's']).max()
        tmp4_more_10.reset_index(inplace=True)
        tmp4_more_10 = tmp4_more_10[tmp4_more_10['ito_his'] <= 10]
        tmp_s_S_list.append(tmp4_more_10)
        tmp4_less_10 = tmp3.groupby(['sku_id', 's']).min()
        tmp4_less_10.reset_index(inplace=True)
        tmp4_less_10 = tmp4_less_10[tmp4_less_10['ito_his'] > 10]
        tmp_s_S_list.append(tmp4_less_10)
    # 如果备选集合中不同s的数量等于1
    else:
        mask = (Selected_s_S['ts_percent'] > 0.8) & (Selected_s_S['ts_percent'] <= 0.98)
        tmp3 = Selected_s_S[mask]
        tmp4_more_10 = tmp3.groupby(['sku_id', 's']).max()
        tmp4_more_10.reset_index(inplace=True)
        tmp4_more_10 = tmp4_more_10[tmp4_more_10['ito_his'] <= 10]
        tmp_s_S_list.append(tmp4_more_10)
        tmp4_less_10 = tmp3.groupby(['sku_id', 's']).min()
        tmp4_less_10.reset_index(inplace=True)
        tmp4_less_10 = tmp4_less_10[tmp4_less_10['ito_his'] > 10]
        tmp_s_S_list.append(tmp4_less_10)
    Selected_s_S_4 = pd.concat(tmp_s_S_list)

'''获取基于现货率和周转的筛选
tmp_kpi=Selected_s_S.groupby(['sku_id']).max()
tmp_kpi.reset_index(inplace=True)
tmp_kpi2=tmp_kpi.loc[:,['sku_id','cr_sim']]
tmp_kpi3=pd.merge(tmp_kpi2,Selected_s_S,left_on=['sku_id','cr_sim'],right_on=['sku_id','cr_sim'])
tmp_kpi4=tmp_kpi3.groupby(['sku_id','ito_sim']).min()
tmp_kpi4.reset_index(inplace=True)
Selected_s_S=pd.merge(tmp_kpi4,Selected_s_S,left_on=['sku_id','cr_sim','ito_sim'],right_on=['sku_id','cr_sim','ito_sim'])
Selected_s_S=Selected_s_S.loc[:,['sku_id','cr_sim','ito_sim','s_y','S_y']]
Selected_s_S.columns=['sku_id','cr_sim','ito_sim','s','S']'''
# '''s min S min'''
# tmp1=Selected_s_S.groupby(['sku_id']).min()
# '''s max S min'''
#
# tmp1=Selected_s_S.groupby(['sku_id']).max()
# tmp1.reset_index(inplace=True)
# tmp2=tmp1.loc[:,['sku_id','s']]
# tmp3=pd.merge(tmp2,Selected_s_S,left_on=['sku_id','s'],right_on=['sku_id','s'])
# '''S的取值'''
# # tmp4=tmp3.groupby(['sku_id','s']).min()
# tmp4=tmp3.groupby(['sku_id','s']).max()
# Selected_s_S=tmp4.reset_index()
# System_s_S=pd.DataFrame({'sku_id':['997186','997160'],
#                          's':[4,5],'S':[10,12]})
# 因为系统参数无法找到固定不变的，所以要采用动态参数，以下部分不方便使用了，注释掉
# System_s_S_mid = pd.read_csv(data_dir0 + para_file_name, sep='\t', dtype=dtype)
# print System_s_S_mid.head()
# System_s_S=System_s_S_mid.loc[:,['sku_id','safestock','maxstock']]
# System_s_S.columns=['sku_id','s','S']
# System_s_S['type']='system'
# all_s_S=pd.concat([Selected_s_S,System_s_S])

Selected_s_S['type'] = 'sim'
if sim_s_S_type == 4:
    Selected_s_S_4['type'] = 'sim'
    all_s_S = Selected_s_S_4
else:
    all_s_S = Selected_s_S
all_s_S.to_csv(path_or_buf=output_dir + train_simulation_name + '_all_s_S.csv', index=False, sep='\t')
##################################################################################################################################
###########################根据仿真训练结果在测试集上进行效果测试#################################################################
##################################################################################################################################
# 保留整体明细数据
system_retail_datasets = []
sim_retail_datasets = []
print u'测试集上进行运行数据'
test_date_range = ['2016-12-16','2016-12-30']
test_sku_kpi = []
test_simulation_results = {}
test_start_dt = datetime.datetime.strptime(test_date_range[0], '%Y-%m-%d')
test_end_dt = datetime.datetime.strptime(test_date_range[1], '%Y-%m-%d')
test_length = (test_end_dt - test_start_dt).days + 1
for sku_id in sku_list:
    # sku_id = '1000054'
    # sku_id = '1107763'            # error 的信息
    # 设置仿真开始日期和0:仿真结束日期
    # 准备数据
    df_sku = df[df.sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.dt < test_date_range[0]) | (df_sku.dt > test_date_range[1])].index)
    # 如果仿真开始日期的销量预测数据为空，不进行仿真
    if df_sku.shape[0] < test_length:
        continue
    if len(df_sku.ofdsales) <= 0:
        logging.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
        continue
    if (not isinstance(df_sku.ofdsales.iloc[0], str)):
        logging.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
        continue
    # 如果仿真开始日期的库存数据为空，不进行仿真
    if np.isnan(df_sku.stock_qtty.iloc[0]):
        logging.info(str(sku_id) + ': ' + u'仿真开始日期库存数据为空，不进行仿真！')
        continue
    # 仿真期间总库存为0，也不仿真
    if abs(np.sum(df_sku.stock_qtty) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + u'仿真期间总库存为0，也不仿真！')
        continue
    # 如果仿真期间总销量为0，不进行仿真
    if abs(np.sum(df_sku.total_sale) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + u'仿真期间总销量为0，不进行仿真！')
        continue
    # sku_name = (df_sku.sku_name.iloc[0]).decode('gbk').encode('utf-8')
    sku_name = ''
    # print(sku_id + '@' + sku_name + u': 开始仿真,关键KPI:sku_id,cr_sim,cr_his,ito_sim,ito_his,ts_sim,ts_his,pur_cnt_sim,s,S.date_range,ito_level')
    logging.info(str(sku_id) + '@' + sku_name + u': 开始仿真......')
    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + s_S_type + u'_origin.csv', index=False, sep='\t')
    sales_his = df_sku.total_sale.as_matrix()
    inv_his = df_sku.stock_qtty.as_matrix()
    sales_per = df_sku.ts_percent.as_matrix()
    cv_sale = df_sku.cv_sale.as_matrix()
    cv_sale = cv_sale[0]
    sales_pred_mean = []
    for x in df_sku.ofdsales:
        if type(x) == float and math.isnan(x):
            sales_pred_mean.append(None)
        else:
            sales_pred_mean.append(ast.literal_eval(x))
            # print u'预测销量相关数据'
            # print sales_pred_mean
    Sku_all_s_S = all_s_S[all_s_S['sku_id'] == sku_id]
    # print Sku_all_s_S
    # print Sku_Selected_s_S,这个针对仿真测试生成的不同的优化参数进行测试不同的
    for index, row in Sku_all_s_S.iterrows():
        s = row['s']
        S = row['S']
        sku_simulation = SkuSimulation(test_date_range, sales_his, inv_his, sales_pred_mean, vlt_val,
                                       vlt_prob, s=s, S=S, sku_id=sku_id, sku_name=sku_name, sale_per=sales_per,
                                       cv_sale=cv_sale)
        sku_simulation.run_simulation(seed=66)
        # 计算相关KPI
        sku_kpi = sku_simulation.calc_kpi()
        sku_kpi.append(row['type'])
        test_sku_kpi.append(sku_kpi)
        test_simulation_results[sku_id] = sku_simulation

        # 将仿真明细数据输出到文件
        if write_daily_data:
            daily_data = sku_simulation.get_daily_data()
            sim_retail_datasets.append(daily_data)
            daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + s_S_type + '_sim.csv', index=False, sep='\t')

        endtime = datetime.datetime.now()
        used_seconds = (endtime - starttime).total_seconds()
    complete_sku += 1

    logging.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                 str(used_seconds))
    '''
    仿真系统的参数
    '''
    # 补货点和补货量的计算分为两种，一种为所有时间的s完全一样，另外一种为s为按天更新的
    s_array = df_sku.safestock.as_matrix()
    S_array = df_sku.maxstock.as_matrix()
    sku_simulation = SkuSimulation(test_date_range, sales_his, inv_his, sales_pred_mean, vlt_val,
                                   vlt_prob, s_array=s_array, S_array=S_array, sku_id=sku_id, sku_name=sku_name,
                                   sale_per=sales_per, cv_sale=cv_sale)
    sku_simulation.run_simulation(seed=66)
    sku_kpi = sku_simulation.calc_kpi()
    sku_kpi.append('system')
    test_sku_kpi.append(sku_kpi)
    test_simulation_results[sku_id] = sku_simulation
    if write_daily_data:
        daily_data = sku_simulation.get_daily_data()
        system_retail_datasets.append(daily_data)
        if np.isnan(sum(daily_data.inv_sim) / sum(daily_data.sales_sim)):
            print sku_id
            daily_data.to_csv(output_dir + 'error_kpi_{0}.csv'.format(sku_id))
            exit()
        daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + s_S_type + '_system.csv', index=False, sep='\t')

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
result_kpi_df = pd.DataFrame.from_records(test_sku_kpi,
                                          columns=['sku_id', 'cr_sim', 'cr_his', 'ito_sim', 'ito_his', 'ts_sim',
                                                   'ts_his', 'pur_cnt_sim', 's', 'S', 'date_begin', 'date_end',
                                                   'ito_level', 'ts_rate', 'ts_percent', 'cv_sale', 'target_s'])
result_kpi_df_sim = result_kpi_df[result_kpi_df['target_s'] == 'sim']
result_kpi_df_system = result_kpi_df[result_kpi_df['target_s'] == 'system']
result_kpi_df = pd.merge(result_kpi_df_sim, result_kpi_df_system, on=['sku_id'], suffixes=('_sim', '_system'))

result_kpi_df = result_kpi_df.loc[:,
                ['sku_id', 'cr_sim_sim', 'cr_sim_system', 'ito_sim_sim', 'ito_sim_system', 's_sim', 'S_sim',
                 'cr_his_sim', 'ito_his_sim', 'ts_sim_sim', 'ts_sim_system', 'ts_his_sim',
                 'cv_sale_system', 'ts_percent_system']]

result_kpi_df.columns = ['sku_id', 'cr_sim', 'cr_system', 'ito_sim', 'ito_system', 's_sim', 'S_sim', 'cr_his',
                         'ito_his', 'ts_sim', 'ts_system',
                         'ts_his', 'cv_sale', 'ts_percent']
result_kpi_df.to_csv(path_or_buf=output_dir + test_simulation_name + s_S_type + '_kpi.csv', index=False, sep='\t')

print s_S_type
print 16 * '*' + u'现货率对比，仿真现货率-系统现货率' + 16 * '*'
print u'现货率提升的SKU数量：', sum((result_kpi_df.cr_sim - result_kpi_df.cr_system) > 0)
print u'现货率下降的SKU数量：', sum((result_kpi_df.cr_sim - result_kpi_df.cr_system) < 0)

print 16 * '*' + u'周转对比，仿真周转-系统周转' + 16 * '*'
print u'周转提升的SKU数量：', sum((result_kpi_df.ito_sim - result_kpi_df.ito_system) > 0)
print u'周转下降的SKU数量：', sum((result_kpi_df.ito_sim - result_kpi_df.ito_system) < 0)

print 16 * '*' + u'周转、现货率综合对比' + 16 * '*'
print u'现货率上升、周转上升的SKU数量：', sum(
    ((result_kpi_df.cr_sim - result_kpi_df.cr_system) > 0) & ((result_kpi_df.ito_sim - result_kpi_df.ito_system) > 0))
print u'现货率上升、周转下降的SKU数量：', sum(
    ((result_kpi_df.cr_sim - result_kpi_df.cr_system) > 0) & ((result_kpi_df.ito_sim - result_kpi_df.ito_system) < 0))

print 16 * '-' + u'计算总体KPI' + 16 * '-'
system_retail_datasets = pd.concat(system_retail_datasets)
sim_retail_datasets = pd.concat(sim_retail_datasets)
system_retail_datasets.to_csv(path_or_buf=output_dir + s_S_type + 'system_retail_datasets.csv', index=False, sep='\t')
sim_retail_datasets.to_csv(path_or_buf=output_dir + s_S_type + 'sim_retail_datasets.csv', index=False, sep='\t')
print 16 * '*' + u'周转对比' + 16 * '*'
print u'系统周转值: ', sum(system_retail_datasets.inv_sim) / sum(system_retail_datasets.sales_sim)
print u'仿真周转值: ', sum(sim_retail_datasets.inv_sim) / sum(sim_retail_datasets.sales_sim)
print 16 * '*' + u'现货率对比' + 16 * '*'
print u'系统现货率值: ', float(sum(system_retail_datasets.inv_sim > 0)) / float(system_retail_datasets.shape[0])
print u'仿真现货率值: ', float(sum(sim_retail_datasets.inv_sim > 0)) / float(sim_retail_datasets.shape[0])

print 16 * '*' + u's,S 频率数据统计报表如下' + 16 * '*'
print u'仿真参数的分布，单个SKU在测试期间的(s,S)保持不变'
df = result_kpi_df.loc[:, ['s_sim', 'S_sim']]
print pd.pivot_table(df, index=['s_sim'], columns=['S_sim'], aggfunc=[len])

print u'系统参数的分布，每天的数值不同，但是相邻今天的(s,S)基本保持不变，统计天次'
df2 = system_retail_datasets.loc[:, ['s', 'S']]
print pd.pivot_table(df2, index=['s'], columns=['S'], aggfunc=[len])

print 16 * '*' + u'基于销量稳定性、销量大小的统计(s,S)分布' + 16 * '*'

result_kpi_df['discrete_cv'] = map(lambda x: 1 if x <= 1 else (2 if 1 < x <= 2 else 3), result_kpi_df['cv_sale'])
result_kpi_df['discrete_per'] = map(
    lambda x: 0.5 if x <= 0.5 else (0.8 if 0.5 < x <= 0.8 else (0.9 if 0.8 < x <= 0.9 else 1)),
    result_kpi_df['ts_percent'])

print u'基于销量大小的统计(s,S)分布'
df2 = result_kpi_df.loc[:, ['s_sim', 'S_sim', 'discrete_per']]
print pd.pivot_table(df2, index=['discrete_per', 's_sim'], columns=['S_sim'], aggfunc=[len])
print u'基于销量稳定性的统计(s,S)分布'
df2 = result_kpi_df.loc[:, ['s_sim', 'S_sim', 'discrete_cv']]
print pd.pivot_table(df2, index=['discrete_cv', 's_sim'], columns=['S_sim'], aggfunc=[len])
print u'基于销量稳定性、销量大小的统计(s,S)分布'
df2 = result_kpi_df.loc[:, ['s_sim', 'S_sim', 'discrete_cv', 'discrete_per']]
print pd.pivot_table(df2, index=['discrete_cv', 'discrete_per', 's_sim'], columns=['S_sim'], aggfunc=[len])
