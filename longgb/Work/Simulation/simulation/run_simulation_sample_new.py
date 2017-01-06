# coding: utf-8

import ast
import datetime
import logging
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from com.jd.pbs.simulation.SkuSimulationNew import SkuSimulation
from com.jd.pbs.simulation.SimulationUtils import SimulationUtils
from com.jd.pbs.simulation.SkuSimulationModifyNew import SkuSimulationPredCorrection
from com.jd.pbs.simulation import config


# 配置信息
data_dir = config.data_dir
data_file_name = config.data_file_name
output_dir = config.output_dir
logging_dir = config.logging_dir

# 本次仿真配置信息：
write_daily_data = True
write_original_data = True
persistence = True
simulation_name = 'simulation_1671'
output_fig = True


logging.basicConfig(filename=logging_dir + simulation_name + '.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# 读入数据
dtype = {'item_sku_id': str, 'ofdsales': str, 'variance': str, 'model': str, 'supp_name': str, 'supp_brevity_cd': str}
df = pd.read_csv(data_dir + data_file_name, sep='\t', dtype=dtype)
df = df.drop(df[df.shelves_dt >= '2016-01-01'].index)

# 获取SKU列表
sku_list = df.item_sku_id.unique()
total_sku = len(sku_list)
complete_sku = 0

sku_kpi = []
simulation_results = {}

starttime = datetime.datetime.now()

counter = 0


for sku_id in sku_list:

    # 设置数据开始日期和结束日期
    date_range = ['2016-01-01', '2016-05-31']

    # 设置仿真开始日期和结束日期
    simulation_date_range = ['2016-01-01', '2016-05-31']

    # 准备数据
    df_sku = df[df.item_sku_id == sku_id]
    df_sku = df_sku.drop(df_sku[(df_sku.day_string < date_range[0]) | (df_sku.day_string > date_range[1])].index)

    if SimulationUtils.check_simulation_condition(df_sku):
        # 不满足仿真条件，跳过该SKU
        continue

    sku_name = (df_sku.sku_name.iloc[0]).decode('gbk').encode('utf-8')
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
    SimulationUtils.fill_nan(cr_pbs)
    bp_pbs = df_sku.buy_period.as_matrix()
    SimulationUtils.fill_nan(bp_pbs)
    lop_pbs = df_sku.lop.as_matrix()
    SimulationUtils.fill_nan(lop_pbs)
    ti_pbs = df_sku.target_inventory.as_matrix()
    SimulationUtils.fill_nan(ti_pbs)
    band = df_sku.band.as_matrix()

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

    sku_simulation = SkuSimulation(
        date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val, vlt_prob, actual_pur_qtty, wh_qtn,
        sku_id=sku_id, sku_name=sku_name, cr_pbs=cr_pbs, bp_pbs=bp_pbs, lop_pbs=lop_pbs, ti_pbs=ti_pbs, band=band)

    # 仿真
    logging.info(str(sku_id) + '@' + 'run_simulation()')
    sku_simulation.run_simulation(cr=0.95, seed=618)
    sku_kpi.append(sku_simulation.calc_kpi())
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
columns = ['sku_id', 'cr_sim', 'cr_his', 'total_inv_sim', 'total_inv_his', 'ito_sim', 'ito_his', 'gmv_sim', 'gmv_his',
           'ts_sim', 'ts_his', 'pur_cnt_sim', 'success_cnt', 'pur_cnt_his', 'wh_qtn', 'avg_bp_pbs']
kpi_df = pd.DataFrame.from_records(sku_kpi, columns=columns)
kpi_df.to_csv(path_or_buf=output_dir + simulation_name + '_kpi.csv', index=False, sep='\t')

# 计算汇总的KPI
kpi_summary = calc_kpi_summary(kpi_df)
print kpi_summary

if persistence:
    pickle.dump(simulation_results, open(output_dir + simulation_name + '.dat', 'wb'), True)
