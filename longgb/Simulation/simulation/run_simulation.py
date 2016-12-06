# coding: utf-8
# ----------------------------------- 路径导入 ---------------------------------------
import os
import sys
from sys import path
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
# test包路径导入
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print pth
import ast
import datetime
import logging
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
sys.path.append(r'D:\Lgb\pythonstudy\longgb')
# from com.jd.pbs.simulation.SkuSimulation import SkuSimulation
# from com.jd.pbs.simulation.SkuSimulationModify import MaxVlt_Times_Demand,SkuSimulationBp25,SkuSimulationMg,SkuSimulationPbs,SkuSimulationSalesCorrection,SkuSimulationSequential,HisSkuBpMeanSimulation
# from com.jd.pbs.simulation import configServer
from simulation.SkuSimulation import SkuSimulation
from simulation.SkuSimulationModify import MaxVlt_Times_Demand,SkuSimulationBp25,SkuSimulationMg,SkuSimulationPbs,SkuSimulationSalesCorrection,HisSkuBpMeanSimulation
# from simulation.SkuSimulationModify import SkuSimulationSequential
from simulation import configServer

import time






time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime())
time.ctime() # 当前时间的字符串形式
time.strptime('2016-11-29_17-57-51', '%Y-%m-%d_%H-%M-%S')
# 配置信息
workingfolderName=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
choice=configServer.start_point(workingfolderName)
strategy = configServer.strategy[choice]
data_dir0 = configServer.data_dir
data_file_name = configServer.setDataFileName()
# data_dir = data_dir0+strategy+'/'
output_dir = configServer.output_dir
# logging_dir = data_dir+configServer.logging_file


# dirFolder = data_dir0+strategy+'/'
workingfolder = data_dir0+strategy+'/'+workingfolderName+'/'
output_dir= workingfolder+output_dir
logging_dir= workingfolder+'log/'



# 本次仿真配置信息：
write_daily_data = True
write_original_data = True
persistence = True
output_fig = True
simulation_name = 'sample_data_base_policy'


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

logging.basicConfig(filename=logging_dir + simulation_name + '.log', level=logging.INFO)

# 读入数据
dtype = {'item_sku_id': str}
df = pd.read_csv(data_dir0 + data_file_name, sep='\t', dtype=dtype)

# 获取SKU列表
sku_list = df.item_sku_id.unique()
total_sku = len(sku_list)
complete_sku = 0

kpi_list = []
simulation_results = {}

starttime = datetime.datetime.now()

for sku_id in sku_list:

    # 设置仿真开始日期和仿真结束日期
    date_range = configServer.date_range

    # 准备数据
    df_sku = df[df.item_sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.day_string < date_range[0]) | (df_sku.day_string > date_range[1])].index)

    # 如果仿真开始日期的销量预测数据为空，不进行仿真
    if (not isinstance(df_sku.ofdsales.iloc[0], str)) or (not isinstance(df_sku.variance.iloc[0], str)):
        logging.info(str(sku_id) + ': ' + '仿真开始日期销量预测数据为空，不进行仿真！')
        continue

    # 如果仿真开始日期的库存数据为空，不进行仿真
    if np.isnan(df_sku.stock_qtty.iloc[0]):
        logging.info(str(sku_id) + ': ' + '仿真开始日期库存数据为空，不进行仿真！')
        continue

    #仿真期间总库存为0，也不仿真
    if abs(np.sum(df_sku.stock_qtty) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + '仿真期间总库存为0，也不仿真！')
        continue

    # 如果仿真期间总销量为0，不进行仿真
    if abs(np.sum(df_sku.total_sales) - 0) <= 1e-3:
        logging.info(str(sku_id) + ': ' + '仿真期间总销量为0，不进行仿真！')
        continue

    # sku_name = (df_sku.sku_name.iloc[0]).decode('gbk').encode('utf-8')
    sku_name = df_sku.sku_name.iloc[0]
    print(sku_id + '@' + sku_name + ': 开始仿真......')
    logging.info(str(sku_id) + '@' + sku_name + ': 开始仿真......')

    # 将原始数据输出到文件
    if write_original_data:
        df_sku.to_csv(path_or_buf=output_dir + str(sku_id) + '_origin.csv', index=False, sep='\t')

    sales_his = df_sku.total_sales.as_matrix()
    inv_his = df_sku.stock_qtty.as_matrix()
    actual_pur_qtty = df_sku.actual_pur_qtty.as_matrix()
    uprc = np.nanmean(df_sku.uprc.as_matrix())
    wh_qtn = np.nanmean(df_sku.wh_qtn.as_matrix())
    cr_pbs = df_sku.cr.as_matrix()
    fill_nan(cr_pbs)
    bp_pbs = df_sku.buy_period.as_matrix()
    fill_nan(bp_pbs)
    lop_pbs = df_sku.lop.as_matrix()
    fill_nan(lop_pbs)
    ti_pbs = df_sku.target_inventory.as_matrix()
    fill_nan(ti_pbs)
    org_nation_sale_num_band = df_sku.org_nation_sale_num_band.iloc[0]
    innerOutQtty = df_sku.inner_outer_qtty.as_matrix()

    sales_pred_mean = []
    for x in df_sku.ofdsales:
        if type(x) == float and math.isnan(x):
            sales_pred_mean.append(None)
        else:
            sales_pred_mean.append(ast.literal_eval(x))

    sales_pred_sd = []
    for x in df_sku.variance:
        if type(x) == float and math.isnan(x):
            sales_pred_sd.append(None)
        else:
            sales_pred_sd.append(ast.literal_eval(x))

    # 计算VLT分布
    # 如果仿真期间没有采购单，默认VLT为10天
    vlt_records = df_sku.vlt.as_matrix()
    vlt_records = vlt_records[~np.isnan(vlt_records)]
    if len(vlt_records) > 0:
        vlt_freq = itemfreq(vlt_records)
        vlt_val = vlt_freq[:, 0]
        vlt_prob = vlt_freq[:, 1] / (vlt_freq[:, 1]).sum()
    else:
        vlt_val = np.array([10])
        vlt_prob = np.array([1.0])

    sku_simulation = SkuSimulation(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                   vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                   cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                   org_nation_sale_num_band=org_nation_sale_num_band)
    if choice == 0:
        pass
    if choice == 1:
        sku_simulation=MaxVlt_Times_Demand(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                             vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                             cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                             org_nation_sale_num_band=org_nation_sale_num_band)
    elif choice ==2:
        sku_simulation=SkuSimulationBp25(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                         vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                         cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                         org_nation_sale_num_band=org_nation_sale_num_band)
    elif choice ==3:
        sku_simulation=SkuSimulationMg(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                       vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                       cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                       org_nation_sale_num_band=org_nation_sale_num_band)
    elif choice ==4:
        sku_simulation=SkuSimulationPbs(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                        vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                        cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                        org_nation_sale_num_band=org_nation_sale_num_band)
    elif choice ==5:
        sku_simulation=SkuSimulationSalesCorrection(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val,
                                                    vlt_prob, actual_pur_qtty, wh_qtn, sku_id=sku_id, sku_name=sku_name,
                                                    cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs,
                                                    org_nation_sale_num_band=org_nation_sale_num_band)

    # 仿真
    logging.info(str(sku_id) + '@' + 'run_simulation()')
    sku_simulation.run_simulation(seed=66)
    print sku_simulation.calc_kpi()
    kpi_list.append(sku_simulation.calc_kpi())
    simulation_results[sku_id] = sku_simulation

    # 将仿真明细数据输出到文件
    if write_daily_data:
        daily_data = sku_simulation.get_daily_data()
        daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '.csv', index=False, sep='\t')

    # 将图片输出到文件
    if output_fig:
        sku_simulation.get_report(output_dir)

    # 运行日志
    endtime = datetime.datetime.now()
    used_seconds = (endtime - starttime).total_seconds()
    complete_sku += 1

    logging.info('Total SKU=' + str(total_sku) + ' | ' + 'Finish SKU=' + str(complete_sku) + ' | ' + 'Used seconds=' +
                 str(used_seconds))

# SKU粒度的KPI
kpi_df = pd.DataFrame.from_records(kpi_list, columns=['sku_id', 'cr_sim', 'cr_his', 'ito_sim', 'ito_his',
                                                      'gmv_sim', 'gmv_his', 'ts_sim', 'ts_his',
                                                      'pur_cnt_sim', 'success_cnt', 'wh_qtn', 'pur_cnt_his',
                                                      'org_nation_sale_num_band'])
kpi_df.to_csv(path_or_buf=output_dir + simulation_name + '_kpi.csv', index=False, sep='\t')

arr=[]
import glob
data = pd.DataFrame()
for name in glob.glob(output_dir+'/*[0-9].csv'):
    # print name
    df = pd.read_csv(name,sep="\\t")
    df=df[["sku_id","dt","sales_his_origin","sales_sim","mean_price","inv_his","inv_sim","sale_num_band"]]
    data = pd.concat([data,df])
cr_his = (data["inv_his"]>0).sum() / float(len(data))
cr_sim = (data["inv_sim"]>0).sum() / float(len(data))
ito_his = np.sum(data["inv_his"]*data["mean_price"]) / float(np.sum(data["sales_his_origin"]*data["mean_price"]))
ito_sim = np.sum(data["inv_sim"]*data["mean_price"]) / float(np.sum(data["sales_sim"]*data["mean_price"]))
gmv_his = np.sum(data["sales_his_origin"] * data["mean_price"])
gmv_sim = np.sum(data["sales_sim"] * data["mean_price"])
total_Kpi = pd.DataFrame(
    {"cr_his":[cr_his],
     "cr_sim":[cr_sim],
     "ito_his":[ito_his],
     "ito_sim":[ito_sim],
     "gmv_his":[gmv_his],
     "gmv_sim":[gmv_sim]
     }
)


for band,grouped in data.groupby("sale_num_band"):
    cr_his_band = (grouped["inv_his"]>0).sum() / float(len(grouped))
    cr_sim_band = (grouped["inv_sim"]>0).sum() / float(len(grouped))
    ito_his_band = np.sum(grouped["inv_his"]*grouped["mean_price"]) / float(np.sum(grouped["sales_his_origin"]*grouped["mean_price"]))
    ito_sim_band = np.sum(grouped["inv_sim"]*grouped["mean_price"]) / float(np.sum(grouped["sales_sim"]*grouped["mean_price"]))
    gmv_his_band = np.sum(grouped["sales_his_origin"] * grouped["mean_price"])
    gmv_sim_band = np.sum(grouped["sales_sim"] * grouped["mean_price"])
    arr.append([band,cr_his_band,cr_sim_band,ito_his_band,ito_sim_band,gmv_his_band,gmv_sim_band])
arr.append(["Total",cr_his,cr_sim,ito_his,ito_sim,gmv_his,gmv_sim])
df= pd.DataFrame(np.array(arr))
df.columns=["band","cr_his_band","cr_sim_band","ito_his_band","ito_sim_band","gmv_his_band","gmv_sim_band"]

df.to_csv(output_dir+"total_Kpi.csv",sep="\t",index=None)
# total_Kpi.to_csv(output_dir+"total_Kpi.csv",sep="\t",index=None)
print "/*************************** Done ********************************/"
print "Please find your kpi report in "+output_dir+"total_Kpi.csv"
print "/*****************************************************************/"
# if persistence:
#     pickle.dump(simulation_results, open(output_dir + simulation_name + '.dat', 'wb'), True)
