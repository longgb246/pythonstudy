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
from inventory_process_online import inventory_proess
import configServer
import time
from StatisUtil import EMsmooth
from collections import defaultdict
import csv
import dill
import  copy
import sys

import  logging

#日志记录部分
# 创建一个logger
logger = logging.getLogger('allocation .. logger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(configServer.log_path)
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler,如果不想在控制台显示，注释掉logger.addHandler(ch)即可
logger.addHandler(fh)
# logger.addHandler(ch)

# 配置信息
workingfolderName=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
data_dir0 = configServer.data_dir
# data_dir = data_dir0+strategy+'/'
output_dir = configServer.output_dir
# logging_dir = data_dir+configServer.logging_file

data_file_name=configServer.data_file_name
rdc_sale_file_name=configServer.rdc_sale_file_name
# dirFolder = data_dir0+strategy+'/'
# logging_dir=configServer.log_path

##根据初始化信息和结束日期变化为list
def datelist(start, end):
    start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(end,'%Y-%m-%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result
# 本次仿真配置信息：
write_daily_data = True
write_original_data = True
## 设置flag值为1 则表示第一次跑，后面运行 直接设置flag为0即可
train_simulation_name = 'train_kpi'
order_flag=0
order_list_file_name='order_list_dict'
rdc_sale_flag=0
rdc_sale_file_dict_name='rdc_sale_list_dict'
rdc_inv_flag=0
rdc_inv_file_dict_name='rdc_list_inv_dict'
fdc_info_flag=0
fdc_info_file_name='fdc_info_file_name_list'
# sys.exit(0)
# Utils
def fill_nan(input_array):
    if isinstance(input_array, np.ndarray):
        for idx in range(len(input_array)):
            if idx == 0 and np.isnan(input_array[idx]):
                input_array[idx] = np.nanmean(input_array)
            elif np.isnan(input_array[idx]):
                input_array[idx] = input_array[idx - 1]
    else:
        print 'Input must be 1d numpy.ndarray!'

# logging.basicConfig(filename=logging_dir + train_simulation_name + '.log', level=logging.INFO)

# 读入数据
dtype = {'sku_id': str}
print u'读入销量相关数据'
df = pd.read_csv(data_dir0 + data_file_name, sep='\t', dtype=dtype)
#将缺失值填充为0
df.fillna(0,inplace=True)
#对销量进行平滑
flag_smooth=2
if flag_smooth==1:
    logger.info(u'开始数据平滑处理')
    df_smooth_pre=df.loc[:,["sku_id","dt","total_sale"]]
    df_smooth_pre.columns=["sku","date","sales"]
    logger.info(u'调用平滑函数')
    df_smooth=EMsmooth.smooth(df_smooth_pre)
    logger.info(u'平滑函数处理完成')
    df["dt"]=df["dt"].astype(str)
    df_smooth["date"]=df_smooth["date"].astype(str)
    df=pd.merge(df,df_smooth,how='inner',left_on=["sku_id","dt"],right_on=["sku","date"])
    df=df.loc[:,["fdc_id","sku_id","ofdsales","stock_qtty","dt","sales_smooth","safestock","maxstock","total_sale_per","av_sale"]]
    df.columns=["fdc_id","sku_id","ofdsales","stock_qtty","dt","total_sale","safestock","maxstock",'ts_percent','cv_sale']
    df.fillna(0, inplace=True)          # 对销量进行平滑后，total_sale会有nan，需要填充0
    df.to_csv(path_or_buf=output_dir +'smooth_datasets.csv', index=False, sep='\t')
    logger.info(u'数据平滑处理完成')
elif flag_smooth==0:
    df = pd.read_csv(output_dir +'smooth_datasets.csv', sep='\t', dtype=dtype)
elif flag_smooth==2:
    # print df.columns
    df=df.loc[:,["dt","fdcid","sku_id","forecast_daily_override_sales","total_sales","stock_qtty","safestock","maxstock","stock_qtty_real","std","white_flag"]]
    df.columns=["dt","fdcid","sku_id","ofdsales","total_sale","stock_qtty_last","safestock","maxstock","stock_qtty","varsales","white_flag"]
    df.fillna(0, inplace=True)          # 对销量进行平滑后，total_sale会有nan，需要填充0
# 获取SKU列表
sku_list = df.sku_id.unique()
# sku_list =['1000125']
#-------------------------------------------------该部分仅用来增加测试数据用，后期删掉--------------------------------------#
#当前测试数据没有标记sku是在某天否为白名单，方便测试增加数据
# df['white_flag']=1 if np.random.rand()>0.01 else 0
# df['varsales']=df['ofdsales']
# df['s']=5
# df['S']=12
#---------------------------------------------------------------------------------------------------------------------------#

total_sku = len(sku_list)
starttime = datetime.datetime.now()

date_end=configServer.date_end
date_start=configServer.date_start
train_date_range = configServer.date_range[0]

# 计算VLT分布，采用历史分布数据
fdc_alt = configServer.vlt_val
fdc_alt_prob = configServer.vlt_prob

#定义索引转换函数
def gene_index(fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s)+':'+str(fdc)+':'+str(sku)
#--------------------------------整体仿真参数-----------------------------------------------------------------------------------#
#RDC采购单数据
#-------------------------------处理采购单数据----------------------------------------------------------------------------------#
if order_flag==1:
    order_file_name=configServer.order_file_name
    logger.info(u'开始读取order数据并转化')
    # pkl_order=open(order_data_path)
    # allocation_order_data=pickle.load(pkl_order)
    # pkl_order.close()
    # print allocation_order_data.head()
    allocation_order_data=pd.read_csv(data_dir0 + order_file_name, sep='\t', dtype=dtype)
    allocation_order_data.columns=['arrive_time','item_sku_id','arrive_quantity','dc_id']
    # allocation_order_data['arrive_quantity']=10
    #print allocation_order_data.head()
    logger.info(u'order数据读取完成')
    logger.info(u'开始处理采购单数据')
    # print allocation_order_data.head()
    tmp_df=allocation_order_data[['arrive_time','item_sku_id','arrive_quantity']]
    # print 'tmp...', tmp_df.head()
    tmp_df.columns=['date','item_sku_id','arrive_quantity']
    tmp_df=tmp_df[tmp_df['date']>date_start]
    order_list=defaultdict(lambda :defaultdict(int))
    # tmp_df=tmp_df.drop_duplicates(subset=['date','item_sku_id'])
    #tmp_dict=tmp_df.set_index(['date','item_sku_id']).unstack(0)['arrive_quantity'].to_dict()
    # for index,row in tmp_df.iterrows():
    #     print index,
    #     print row['date']
    #     print row['item_sku_id']
    #     print row['arrive_quantity']
    logger.info(u'进行字典推导更新')
    # order_list.update({row['date']:{row['item_sku_id']:row['arrive_quantity']}
    #                    for index,row in tmp_df.iterrows()})
    for index,row in tmp_df.iterrows():
        if order_list.has_key(row['date']):
            if order_list[row['date']].has_key(str(row['item_sku_id'])):
                order_list[row['date']][str(row['item_sku_id'])]=order_list[row['date']][str(row['item_sku_id'])]+row['arrive_quantity']
            else:
                order_list[row['date']][str(row['item_sku_id'])]=row['arrive_quantity']
        else:
            order_list[row['date']]={str(row['item_sku_id']):row['arrive_quantity']}
    filename=open(data_dir0 + order_list_file_name,'wb')
    pickle.dump(order_list,filename)
    filename.close()
    logger.info(u'order字典更新完成')
    logger.info(u'遍历中间字典，更新采购单字典')
else:
     filename=open(data_dir0 + order_list_file_name,'rb')
     order_list=pickle.load(filename)
     filename.close()
##########更新增加RDC销量###################################3
logger.info(u'开始加工rdc销量数据')
if rdc_sale_flag==1:
    rdc_sale_list=defaultdict(lambda :defaultdict(int))
    allocation_rdc_sale_data=pd.read_csv(data_dir0 + rdc_sale_file_name, sep='\t', dtype=dtype)
    tmp_allocation_rdc_sale_data=allocation_rdc_sale_data.loc[:,['sku_id','dt','total_sales']]
    tmp_allocation_rdc_sale_data=tmp_allocation_rdc_sale_data[tmp_allocation_rdc_sale_data['dt']>=date_start]
    for index,row in tmp_allocation_rdc_sale_data.iterrows():
        if rdc_sale_list.has_key(row['dt']):
            if rdc_sale_list[row['dt']].has_key(row['sku_id']):
                rdc_sale_list[row['dt']][row['sku_id']]=rdc_sale_list[row['dt']][row['sku_id']]+row['total_sales']
            else:
                rdc_sale_list[row['dt']][row['sku_id']]=row['total_sales']
        else:
            rdc_sale_list[row['dt']]={row['sku_id']:row['total_sales']}
    filename=open(data_dir0 + rdc_sale_file_dict_name,'wb')
    pickle.dump(rdc_sale_list,filename)
    filename.close()
else:
    filename=open(data_dir0 + rdc_sale_file_dict_name,'rb')
    rdc_sale_list=pickle.load(filename)
    filename.close()
logger.info(u'rdc销量数据加工完成')
# for k,v in tmp_dict.items():
#     #print k
#     #print '----------------------------------------'
#     for k1,v1 in v.items():
#         #print k1
#         #print '##########################################'
#         #print v1
#         if v1!=None:
#             order_list[k][k1]=v1
logger.info(u'采购单数据处理完成')
#--------------------------------------------------------RDC库存数据-------------------------------------------------------#
#RDC库存数据,获取RDC初始库存即可
logger.info(u'加工处理rdc库存数据')
if rdc_inv_flag==1:
    rdc_inv_file_name=configServer.rdc_inv_file_name
    rdc_inv=defaultdict(int)
    allocation_rdc_data=pd.read_csv(data_dir0 + rdc_inv_file_name, sep='\t', dtype=dtype)
    allocation_rdc_data=allocation_rdc_data[allocation_rdc_data['dt']>=date_start]
    for kk,row in allocation_rdc_data.iterrows():
        index_rdc = gene_index('rdc', row['sku_id'], row['dt'])
        rdc_inv[index_rdc]=row['stock_qtty']
    filename=open(data_dir0 + rdc_inv_file_dict_name,'wb')
    pickle.dump(rdc_inv,filename)
    filename.close()
else:
    filename=open(data_dir0 + rdc_inv_file_dict_name,'rb')
    rdc_inv=pickle.load(filename)
    filename.close()
logger.info(u'加工处理rdc库存数据完成')
#-------------------------------------------------------补充其他变量-------------------------------------------------------#
# 7，LOP-Inverse模型：模型参数与LOP-std-7模型一致，更改了FDC顺序，将重庆与贵阳对换。
fdc_list=[634,633,605]
# fdc_list=[605,633,634]
mid_date_range=configServer.date_range[1]
date_range=datelist(mid_date_range[0],mid_date_range[1])
save_data_path=configServer.output_dir
##################################################################################################################################
###########################根据仿真训练结果在测试集上进行效果测试#################################################################
##################################################################################################################################
#保留整体明细数据，system_retail_datasets系统参数的明细，sim_retail_datasets补货点逻辑的结果
system_retail_datasets=[]
sim_retail_datasets=[]

#记录保存RDC库存数据
system_rdc_datasets=[]
sim_rdc_datasets=[]

test_date_range=configServer.date_range[1]
#KPI结果，sim_fdc_sku_kpi 补货点逻辑的KPI，system_fdc_sku_kpi系统参数的KPI
sim_fdc_sku_kpi=[]
system_fdc_sku_kpi=[]

#数据保留标记 sim表示补货点逻辑的数据，system表示系统参数的数据
sim_lable='sim'
system_label='system'

test_start_dt = datetime.datetime.strptime(test_date_range[0], '%Y-%m-%d')
test_end_dt = datetime.datetime.strptime(test_date_range[1], '%Y-%m-%d')
test_length=(test_end_dt-test_start_dt).days + 1
complete_sku=0
logger.info(u'开始进行仿真程序')
#创建一个字典记录各种SKU的情况
sku_return_result=defaultdict(int)
# l
##如果SKU已经存在结果，就不重复跑了
# sku_complete = map(lambda  x:x.split('_')[0],os.listdir(output_dir))
# print sku_complete
##
# for sku_id in sku_list:
# 设置仿真开始日期和0:仿真结束日期
# 准备数据
# print sku_id
# if sku_id in sku_complete:
# #     continue
# logger.info(u'开始仿真SKU'+str(sku_id))
df_sku = df
df_sku = df_sku.drop(df_sku[(df_sku.dt <test_date_range[0]) | (df_sku.dt > test_date_range[1])].index)
# # 如果仿真开始日期的销量预测数据为空，不进行仿真
# # print df_sku.shape[0]
# # print len(df_sku.ofdsales)
# # print np.isnan(df_sku.stock_qtty.iloc[0])
# # print abs(np.sum(df_sku.stock_qtty) - 0)
# # print abs(np.sum(df_sku.total_sale) - 0)
# if df_sku.shape[0]<test_length:
#     continue
# if len(df_sku.ofdsales)<=0:
#     logger.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
#     continue
# if (not isinstance(df_sku.ofdsales.iloc[0], str)):
#     logger.info(str(sku_id) + ': ' + u'仿真开始日期销量预测数据为空，不进行仿真！')
#     sku_return_result['forecastisnull']+=1
#     continue
# logger.info(u'开始仿真SKU'+str(sku_id)+'-----------------')
# # 如果仿真开始日期的库存数据为空，不进行仿真
# if np.isnan(df_sku.stock_qtty.iloc[0]):
#     logger.info(str(sku_id) + ': ' + u'仿真开始日期库存数据为空，不进行仿真！')
#     sku_return_result['invbeginisnull']+=1
#     continue
#
# # #仿真期间总库存为0，也不仿真
# # if abs(np.sum(df_sku.stock_qtty) - 0) <= 1e-3:
# #     logger.info(str(sku_id) + ': ' + u'仿真期间总库存为0，也不仿真！')
# #     continue
#
# # 如果仿真期间总销量为0，不进行仿真
# if abs(np.sum(df_sku.total_sale) - 0) <= 1e-3:
#     logger.info(str(sku_id) + ': ' + u'仿真期间总销量为0，不进行仿真！')
#     sku_return_result['totalsalesis0']+=1
#     continue
# sku_return_result['simsucess']+=1
# # 将原始数据输出到文件
# if write_original_data:
#     df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + '_origin.csv', index=False, sep='\t')
# sku_name = ''
# # print(sku_id + '@' + sku_name + u': 开始仿真,关键KPI:sku_id,cr_sim,cr_his,ito_sim,ito_his,ts_sim,ts_his,pur_cnt_sim,s,S.date_range,ito_level')
# logger.info(str(sku_id) + '@' + sku_name + u': 开始仿真......')
#
# # 将原始数据输出到文件
# if write_original_data:
#     df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + u'_origin.csv', index=False, sep='\t')
# #保存预测数据
fdc_forecast_sales=defaultdict(list)
fdc_forecast_std=defaultdict(list)
fdc_inv=defaultdict(lambda :defaultdict(int))
white_flag=defaultdict(int)
sales_retail=defaultdict(int)
fdc_his_inv=defaultdict(int)
system_small_s=defaultdict(int)
system_bigger_S=defaultdict(int)
print len(np.unique(df_sku.sku_id))
if fdc_info_flag==1:
    for ky,row in df_sku.iterrows():
        # if row['fdcid'] not in fdc_list:
        #     continue
        # print row['fdcid']
        index=gene_index(row['fdcid'],row['sku_id'],row['dt'])
        #加入预测均值和预测标准差
        # print row
        # print row['ofdsales']
        # print type(row['ofdsales'])
        if type(row['ofdsales']) == float and math.isnan(row['ofdsales']):
            fdc_forecast_sales[index].append(None)
        elif type(row['ofdsales']) == int:
            fdc_forecast_sales[index].extend([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        else:
            fdc_forecast_sales[index].extend(ast.literal_eval(row['ofdsales']))
        #加入预测标准差
        # if type(row['varsales']) == float and math.isnan(row['varsales']):
        #     fdc_forecast_std[index].append(None)
        # else:
        #     fdc_forecast_std[index].extend(ast.literal_eval(row['varsales']))
        fdc_inv[index]['inv']=row['stock_qtty']
        # white_flag[index]=row['white_flag']
        sales_retail[index]=row['total_sale']
        fdc_his_inv[index] = row['stock_qtty']
        fdc_forecast_std[index]=row['varsales']
        white_flag[index]=row['white_flag']
        #如果运行系统调拨量，需要用到s,S
        system_small_s[index]=row["safestock"]
        system_bigger_S[index]=row["maxstock"]
    # print fdc_inv
    # print fdc_his_inv
    #保存转换后的字典
    # #保存预测数据
    fdc_info_list=[fdc_forecast_sales,fdc_forecast_std,fdc_inv,white_flag,sales_retail,fdc_his_inv,system_small_s,system_bigger_S]
    filename=open(data_dir0 + fdc_info_file_name,'wb')
    pickle.dump(fdc_info_list,filename)
    filename.close()
    logger.info(u'fdc all  info 字典更新完成')
else:
    filename=open(data_dir0 + fdc_info_file_name,'rb')
    fdc_iinfo_list=pickle.load(filename)
    filename.close()
    fdc_forecast_sales=fdc_iinfo_list[0]
    fdc_forecast_std=fdc_iinfo_list[1]
    fdc_inv=fdc_iinfo_list[2]
    white_flag=fdc_iinfo_list[3]
    sales_retail=fdc_iinfo_list[4]
    fdc_his_inv=fdc_iinfo_list[5]
    system_small_s=fdc_iinfo_list[6]
    system_bigger_S=fdc_iinfo_list[7]
#传入的为引用，所以需要对起进行深度copy,后续system仿真使用,主要是需要进行读写操作的类型
fdc_inv_system=copy.deepcopy(fdc_inv)
#按照公式计算补货点和补货量，system_flag=0
fdc_allocation=inventory_proess(sku_list=sku_list,fdc_forecast_sales=fdc_forecast_sales, fdc_forecast_std=fdc_forecast_std,
                 fdc_alt=fdc_alt, fdc_alt_prob=fdc_alt_prob, fdc_inv=fdc_inv, white_flag=white_flag,
                 fdc_list=fdc_list, rdc_inv=rdc_inv, date_range=date_range, sales_retail=sales_retail,
                 order_list=order_list, fdc_his_inv=fdc_his_inv,system_small_s=system_small_s,
                system_bigger_S=system_bigger_S,system_flag=0,rdc_sale_list=rdc_sale_list,
                 logger=logger,save_data_path=save_data_path)
fdc_allocation.allocationSimulation()
# #按照SKU保存数据
# if write_daily_data:
#     daily_data_01,daily_data_02 = fdc_allocation.get_daily_data()
#     daily_data_02.to_csv(path_or_buf=output_dir + str(sku_id) +'_'+sim_lable+'.csv', index=False, sep='\t')
#     sim_retail_datasets.append(daily_data_02)
#     sim_rdc_datasets.append(daily_data_01)
#KPI计算函数，计算出各个FDC的SKU对应的KPI
# kpi_result_sku_fdc=fdc_allocation.calc_kpi()
# kpi_result_sku_fdc.to_csv(path_or_buf=output_dir + str(sku_id) +'_'+sim_lable+'_kpi.csv', index=False, sep='\t')
# sim_fdc_sku_kpi.append(kpi_result_sku_fdc)

#按照SKU保存数据
# if write_daily_data:
#     daily_data_01,daily_data_02 = fdc_allocation.get_daily_data()
#     daily_data_02.to_csv(path_or_buf=output_dir + str(sku_id) +'_'+system_label+'.csv', index=False, sep='\t')
#     system_retail_datasets.append(daily_data_02)
#     system_rdc_datasets.append(daily_data_01)
#KPI计算函数，计算出各个FDC的SKU对应的KPI
# kpi_result_sku_fdc=fdc_allocation.calc_kpi()
# kpi_result_sku_fdc.to_csv(path_or_buf=output_dir + str(sku_id)+'_' +system_label+'_kpi.csv', index=False, sep='\t')
# system_fdc_sku_kpi.append(kpi_result_sku_fdc)


#保留补货点计算的补货参数

#处理格式化FDC相关数据
rdc_mid_inv, sim_retail_df = fdc_allocation.get_daily_data()
print u'...计算的数据结构....'
print rdc_mid_inv.shape
print sim_retail_df.shape

sim_retail_df.fillna(0,inplace=True)
sim_retail_df.reset_index(inplace=True)
daily_data_mid2=pd.DataFrame(list(sim_retail_df['index'].apply(lambda x:x.split(':'),1)))
daily_data_mid2.columns=['dt','fdc_id','sku_id']
# print 'fdc_id',pd.unique(daily_data_mid2.fdc_id)
del sim_retail_df['index']
sim_reuslt_daily_data=daily_data_mid2.join(sim_retail_df)
sim_reuslt_daily_data = sim_reuslt_daily_data[sim_reuslt_daily_data['fdc_id'] != 'rdc']

#处理，格式化RDC库存
sim_rdc_df=rdc_mid_inv
rdc_mid2_inv=pd.DataFrame(list(sim_rdc_df['index'].apply(lambda x:x.split(':'),1)))
rdc_mid2_inv.columns=['dt','rdc','sku_id']
del sim_rdc_df['index']
sim_rdc_inv_df=rdc_mid2_inv.join(sim_rdc_df)
sim_rdc_inv_df=sim_rdc_inv_df.loc[:,['dt','sku_id','rdc_inv']]
#将数据存入指定路径
sim_reuslt_daily_data=pd.merge(sim_reuslt_daily_data,sim_rdc_inv_df,on=['dt','sku_id'],how='left')
sim_reuslt_daily_data=sim_reuslt_daily_data[sim_reuslt_daily_data['fdc_id']!='rdc']
sim_reuslt_daily_data.to_csv(path_or_buf=output_dir +sim_lable+'_all_sku_retail.csv', index=False, sep='\t')
# assert 1==2
sim_reuslt_daily_data=sim_reuslt_daily_data[sim_reuslt_daily_data.dt<=date_end]
#计算各个KPI
for fdcsku,fdcdata in sim_reuslt_daily_data.groupby(['fdc_id','sku_id']):
    tmp_fdcid,sku_id=fdcsku[0],fdcsku[1]
    fdc_kpi=defaultdict(lambda :defaultdict(float))
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid]=sum(fdcdata.inv_his>0)/float(len(fdcdata.inv_his))
        fdc_kpi['cr_sim'][tmp_fdcid]=sum(fdcdata.inv_sim>0)/float(len(fdcdata.inv_sim))
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim))<=0 else float(np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin))<=0 else float(np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid]=-1 if float(fdc_kpi['ts_his'][tmp_fdcid])<=0 else float(fdc_kpi['ts_sim'][tmp_fdcid])/float(fdc_kpi['ts_his'][tmp_fdcid])
    tmp_mid_kpi=pd.DataFrame(fdc_kpi)
    tmp_mid_kpi.reset_index(inplace=True)
    tmp_mid_kpi.rename(columns={'index':'fdc_id'},inplace=True)
    tmp_mid_kpi['sku_id']=sku_id
    sim_fdc_sku_kpi.append(tmp_mid_kpi)
sim_result_kpi_df = pd.concat(sim_fdc_sku_kpi)
sim_result_kpi_df.to_csv(path_or_buf=output_dir +sim_lable+'_all_sku_kpi.csv', index=False, sep='\t')

#按照系统参数进行计算，system_flag=1
fdc_allocation=inventory_proess(sku_list=sku_list,fdc_forecast_sales=fdc_forecast_sales, fdc_forecast_std=fdc_forecast_std,
                                fdc_alt=fdc_alt, fdc_alt_prob=fdc_alt_prob, fdc_inv=fdc_inv_system, white_flag=white_flag,
                                fdc_list=fdc_list, rdc_inv=rdc_inv, date_range=date_range, sales_retail=sales_retail,
                                order_list=order_list, fdc_his_inv=fdc_his_inv,system_small_s=system_small_s,
                                system_bigger_S=system_bigger_S,system_flag=1,rdc_sale_list=rdc_sale_list,
                                logger=logger,save_data_path=save_data_path)
fdc_allocation.allocationSimulation()
rdc_mid_inv, system_retail_df = fdc_allocation.get_daily_data()
#系统参数数据处理
system_retail_df=system_retail_df
system_retail_df.fillna(0,inplace=True)
system_retail_df.reset_index(inplace=True)
daily_data_mid2=pd.DataFrame(list(system_retail_df['index'].apply(lambda x:x.split(':'),1)))
daily_data_mid2.columns=['dt','fdc_id','sku_id']
# print 'fdc_id',pd.unique(daily_data_mid2.fdc_id)
del system_retail_df['index']
system_reuslt_daily_data=daily_data_mid2.join(system_retail_df)
#处理，格式化RDC库存
system_rdc_df=rdc_mid_inv
rdc_mid2_inv=pd.DataFrame(list(system_rdc_df['index'].apply(lambda x:x.split(':'),1)))
rdc_mid2_inv.columns=['dt','rdc','sku_id']
del system_rdc_df['index']
system_rdc_inv_df=rdc_mid2_inv.join(system_rdc_df)
system_rdc_inv_df=system_rdc_inv_df.loc[:,['dt','sku_id','rdc_inv']]
#将数据存入指定路径
system_reuslt_daily_data=pd.merge(system_reuslt_daily_data,system_rdc_inv_df,on=['dt','sku_id'],how='left')
system_reuslt_daily_data=system_reuslt_daily_data[system_reuslt_daily_data['fdc_id']!='rdc']
system_reuslt_daily_data.to_csv(path_or_buf=output_dir +system_label+'_all_sku_retail.csv', index=False, sep='\t')
#计算各个SKU的KPI
system_reuslt_daily_data=system_reuslt_daily_data[system_reuslt_daily_data.dt<=date_end]
for fdcsku,fdcdata in system_reuslt_daily_data.groupby(['fdc_id','sku_id']):
    tmp_fdcid,sku_id=fdcsku[0],fdcsku[1]
    # print tmp_fdcid,sku_id
    fdc_kpi=defaultdict(lambda :defaultdict(float))
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid]=sum(fdcdata.inv_his>0)/float(len(fdcdata.inv_his))
        fdc_kpi['cr_sim'][tmp_fdcid]=sum(fdcdata.inv_sim>0)/float(len(fdcdata.inv_sim))
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim))<=0 else float(np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin))<=0 else float(np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid]=-1 if float(fdc_kpi['ts_his'][tmp_fdcid])<=0 else float(fdc_kpi['ts_sim'][tmp_fdcid])/float(fdc_kpi['ts_his'][tmp_fdcid])
    tmp_mid_kpi=pd.DataFrame(fdc_kpi)
    tmp_mid_kpi.reset_index(inplace=True)
    tmp_mid_kpi.rename(columns={'index':'fdc_id'},inplace=True)
    tmp_mid_kpi['sku_id']=sku_id
    system_fdc_sku_kpi.append(tmp_mid_kpi)
system_result_kpi_df = pd.concat(system_fdc_sku_kpi)
system_result_kpi_df.to_csv(path_or_buf=output_dir +system_label+'_all_sku_kpi.csv', index=False, sep='\t')


sku_cnt=len(np.unique(sim_reuslt_daily_data.sku_id))

#计算补货点计算的KPI
fdc_kpi=defaultdict(lambda :defaultdict(float))
for tmp_fdcid,fdcdata in sim_reuslt_daily_data.groupby(['fdc_id']):
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid]=sum(fdcdata.inv_his>0)/float(len(fdcdata.sku_id))
        fdc_kpi['cr_sim'][tmp_fdcid]=sum(fdcdata.inv_sim>0)/float(len(fdcdata.sku_id))
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim))<=0 else float(np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin))<=0 else float(np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid]=-1 if float(fdc_kpi['ts_his'][tmp_fdcid])<=0 else float(fdc_kpi['ts_sim'][tmp_fdcid])/float(fdc_kpi['ts_his'][tmp_fdcid])
sim_fdc_kpi=pd.DataFrame(fdc_kpi)
sim_fdc_kpi.reset_index(inplace=True)
sim_fdc_kpi.rename(columns={'index':'fdc_id'},inplace=True)
#计算系统的KPI
sku_cnt=len(np.unique(sim_reuslt_daily_data.sku_id))
fdc_kpi=defaultdict(lambda :defaultdict(float))
for tmp_fdcid,fdcdata in system_reuslt_daily_data.groupby(['fdc_id']):
    if 'rdc' not in tmp_fdcid:
        # 现货率（cr）：有货天数除以总天数
        fdc_kpi['cr_his'][tmp_fdcid]=sum(fdcdata.inv_his>0)/float(len(date_range)*sku_cnt)
        fdc_kpi['cr_sim'][tmp_fdcid]=sum(fdcdata.inv_sim>0)/float(len(date_range)*sku_cnt)
        # 周转天数（ito）：平均库存除以平均销量
        fdc_kpi['ito_sim'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.fdc_sales_sim))<=0 else float(np.nanmean(fdcdata.inv_sim)) / float(np.nanmean(fdcdata.fdc_sales_sim))
        fdc_kpi['ito_his'][tmp_fdcid] = -1 if float(np.nanmean(fdcdata.sales_his_origin))<=0 else float(np.nanmean(fdcdata.inv_his)) / float(np.nanmean(fdcdata.sales_his_origin))
        # 总销量（ts）
        fdc_kpi['ts_sim'][tmp_fdcid] = np.sum(fdcdata.fdc_sales_sim)
        fdc_kpi['ts_his'][tmp_fdcid] = np.sum(fdcdata.sales_his_origin)
        fdc_kpi['ts_rate'][tmp_fdcid]=-1 if float(fdc_kpi['ts_his'][tmp_fdcid])<=0 else float(fdc_kpi['ts_sim'][tmp_fdcid])/float(fdc_kpi['ts_his'][tmp_fdcid])
system_fdc_kpi=pd.DataFrame(fdc_kpi)
system_fdc_kpi.reset_index(inplace=True)
system_fdc_kpi.rename(columns={'index':'fdc_id'},inplace=True)
fdc_kpi=pd.merge(sim_fdc_kpi,system_fdc_kpi,on=['fdc_id'],suffixes=['_sim','_system'])
fdc_kpi.to_csv(path_or_buf=output_dir+'fdc_kpi.csv', index=False, sep='\t')
# print sku_return_result