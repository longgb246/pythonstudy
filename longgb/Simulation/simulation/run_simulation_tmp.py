# coding: utf-8

import ast
import math
import datetime
import logging
import pandas as pd
import numpy as np
from scipy.stats import itemfreq
import sys
sys.path.append("/home/cmo_ipc/stockPlan/ipc_inv_opt/src/")
import config, SkuSimulation


# 配置信息
data_dir = config.data_dir
data_file_name = config.data_file_name
output_dir = config.output_dir
logging_file = config.logging_file

logging.basicConfig(filename=logging_file, level=logging.INFO)

# 读入数据
dtype = {'item_sku_id': str}
df = pd.read_csv(data_dir + data_file_name, sep='\t', dtype=dtype)

# 获取SKU列表
sku_list = df.item_sku_id.unique()

complete_sku = 0
total_sku = len(sku_list)
starttime = datetime.datetime.now()

for sku_id in sku_list:

    # 准备数据
    df_sku = df[df.item_sku_id == sku_id]
    sku_name = (df_sku.sku_name.iloc[0]).decode('utf-8')
    print(sku_id + '@' + sku_name)

    # 设置仿真开始日期和仿真结束日期
    date_range = ['2016-01-01', '2016-10-24']
    df_sku = df_sku.drop(df_sku[(df_sku.day_string < date_range[0]) | (df_sku.day_string > date_range[1])].index)

    sales_his = df_sku.total_sales.as_matrix()
    inv_his = df_sku.stock_qtty.as_matrix()

    # 如果仿真开始日期的销量预测数据不为空，进行仿真
    if isinstance(df_sku.ofdsales.iloc[0], str) and isinstance(df_sku.variance.iloc[0], str):
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

        actual_pur_qtty = df_sku.actual_pur_qtty.as_matrix()
        uprc = np.mean(df_sku.uprc)
        lop_pbs = df_sku.lop.as_matrix()
        cr = 0.95

        # 实例化仿真类
        sku_simulation = SkuSimulation.SkuSimulationBp25(date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd,
                                                         vlt_val, vlt_prob, cr, actual_pur_qtty, uprc, lop_pbs=lop_pbs,
                                                         sku_id=sku_id, sku_name=sku_name)

        # 仿真
        logging.info(str(sku_id) + '@' + 'run_simulation()')
        sku_simulation.run_simulation()

        # 将仿真明细数据输出到文件
        daily_data = sku_simulation.get_daily_data()
        daily_data.to_csv(path_or_buf=output_dir + str(sku_id) + '.csv', index=False, sep='\t')

    # 运行日志
    endtime = datetime.datetime.now()
    used_seconds = (endtime - starttime).total_seconds()
    complete_sku += 1
    logging.info(str(sku_id) + '@' + sku_name)
    logging.info('Total SKU=' + str(total_sku))
    logging.info('Finish SKU=' + str(complete_sku))
    logging.info('Used seconds=' + str(used_seconds))
