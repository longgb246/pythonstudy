# coding: utf-8

import math
import ast
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from com.jd.pbs.simulation.SkuSimulationNew import SkuSimulation
from com.jd.pbs.simulation.SimulationUtils import SimulationUtils
from com.jd.pbs.simulation.SkuSimulationModifyNew import SkuSimulationPredCorrection


class SimulationUtils:

    def __init__(self):
        pass

    @staticmethod
    def fill_nan(input_array):
        if isinstance(input_array, np.ndarray):
            for idx in range(len(input_array)):
                if idx == 0 and np.isnan(input_array[idx]):
                    input_array[idx] = np.nanmean(input_array)
                elif np.isnan(input_array[idx]):
                    input_array[idx] = input_array[idx - 1]
        else:
            print 'Input must be 1d numpy.ndarray!'

    @staticmethod
    def calc_summary_kpi(df, simulation_name):
        avg_cr_sim = np.mean(df.cr_sim)
        avg_cr_his = np.mean(df.cr_his)
        avg_ito_amount_sim = np.sum(df.total_inv_sim * df.wh_qtn) / np.sum(df.gmv_sim)
        avg_ito_amount_his = np.sum(df.total_inv_his * df.wh_qtn) / np.sum(df.gmv_his)
        avg_ito_sim = np.mean(df.ito_sim)
        avg_ito_his = np.mean(df.ito_his)
        rec = [
            [simulation_name, avg_cr_his, avg_cr_sim, avg_ito_amount_sim, avg_ito_amount_his, avg_ito_sim, avg_ito_his]]
        col = ['simulation_name', 'avg_cr_his', 'avg_cr_sim', 'avg_ito_amount_sim', 'avg_ito_amount_his', 'avg_ito_sim',
               'avg_ito_his']
        kpi_summary_df = pd.DataFrame.from_records(rec, columns=col)
        return kpi_summary_df

    @staticmethod
    def check_simulation_condition(pd_df):
        # 如果仿真开始日期的销量预测数据为空，不进行仿真
        flag1 = (not isinstance(pd_df.ofdsales.iloc[0], str)) or (not isinstance(pd_df.variance.iloc[0], str))
        if flag1:
            return True

        # 如果仿真开始日期的库存数据为空，不进行仿真
        flag2 = np.isnan(pd_df.stock_qtty.iloc[0])
        if flag2:
            return True

        # 如果仿真期间总销量为0，不进行仿真
        flag3 = abs(np.sum(pd_df.total_sales) - 0) <= 1e-3
        if flag3:
            return True

        # 如果仿真期间库存全部为0，不进行仿真
        flag4 = abs(np.sum(pd_df.stock_qtty) - 0) <= 1e-3
        if flag4:
            return True

    @staticmethod
    def get_simulation_instance(df_sku):

        sales_his = df_sku.total_sales.as_matrix()
        inv_his = df_sku.stock_qtty.as_matrix()
        actual_pur_qtty = df_sku.actual_pur_qtty.as_matrix()
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

        return sku_simulation
