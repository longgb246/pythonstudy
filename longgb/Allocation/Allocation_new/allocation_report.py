#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os

# =======================================================================
# =                                 路径设置                             =
# =======================================================================
# 总路径
save_path = r'D:\Lgb\ReadData'          # 总/analysis
path_actual = r'D:\Lgb\ReadData'        # 总
path_sim = r'D:\Lgb\rz_sz'              # 总/analysis
# KPI 实际数据
fill_actual_path = path_actual + os.sep + 'full_actual.xls'
ito_actual_path = path_actual + os.sep + 'ito_actual.xls'
# KPI 仿真数据
table_fill_path = path_sim + os.sep + 'table_fill.csv'
table_ito_path = path_sim + os.sep + 'table_ito.csv'
# Sku 实际数据
ito_sku_nodate_actual_path = path_actual + os.sep + 'sku_ito'
# Sku 仿真数据
table_ito_sku_nodate_path = path_sim + os.sep + 'table_ito_sku_nodate.csv'


if __name__ == '__main__':
    # ===============================================================================
    # =                                 （1）Kpi 数据表                             =
    # ===============================================================================
    # 1、读取数据
    # 实际数据
    fill_actual = pd.read_excel(fill_actual_path)
    fill_actual.columns = ['fdc', 'date_s', 'rdc_order_num_actual', 'total_order_num_actual', 'satisfy_rate_actual']
    fill_actual['fdc_order_num_actual'] = fill_actual['total_order_num_actual'] - fill_actual['rdc_order_num_actual']
    fill_actual_need = fill_actual.loc[:,['fdc', 'date_s', 'fdc_order_num_actual', 'total_order_num_actual', 'satisfy_rate_actual']]
    ito_actual = pd.read_excel(ito_actual_path, sheetname='Sheet1')
    ito_actual.columns = ['fdc', 'date_s', 'inv_actual', 'sale_qtty_actual', 'ito_actual']
    # 仿真数据
    table_fill = pd.read_csv(table_fill_path)
    table_fill.columns = ['fdc', 'date_s', 'fdc_order_num_sim', 'total_order_num_sim', 'fill_rate_sim']
    table_ito = pd.read_csv(table_ito_path)
    table_ito.columns = ['date_s', 'fdc', 'inv_sim', 'sale_qtty_sim', 'ito_rate_sim']

    # 2、合并表 Kpi_report
    Kpi_report = fill_actual_need.merge(table_fill, on=['fdc', 'date_s'])
    Kpi_report['loss_order'] = Kpi_report['total_order_num_actual'] - Kpi_report['total_order_num_sim']
    Kpi_report['loss_order_rate'] = Kpi_report['loss_order'] * 1.0 / Kpi_report['total_order_num_actual']
    Kpi_report = Kpi_report.merge(ito_actual, on=['fdc', 'date_s'])
    Kpi_report = Kpi_report.merge(table_ito, on=['fdc', 'date_s'])

    # 3、保存报表
    Kpi_report.to_csv(save_path + os.sep + 'Kpi_report.csv')


    # ===============================================================================
    # =                                 （1）Sku 粒度表                              =
    # ===============================================================================
    # 1、读取数据
    # 仿真数据
    ito_sku_nodate_actual = pd.read_table(ito_sku_nodate_actual_path, header=None)
    ito_sku_nodate_actual.columns = ['fdc', 'sku', 'inv_actual', 'sale_qtty_actual', 'ito_rate_actual']
    ito_sku_nodate_actual['fdc'] = ito_sku_nodate_actual['fdc'].astype(str)
    ito_sku_nodate_actual['sku'] = ito_sku_nodate_actual['sku'].astype(str)
    # 实际数据
    table_ito_sku_nodate = pd.read_csv(table_ito_sku_nodate_path)
    table_ito_sku_nodate.columns = ['sku', 'fdc', 'inv_sim', 'sale_qtty_sim', 'ito_rate_sim']
    table_ito_sku_nodate['fdc'] = table_ito_sku_nodate['fdc'].astype(str)
    table_ito_sku_nodate['sku'] = table_ito_sku_nodate['sku'].astype(str)

    # 2、合并表 Sku_report
    Sku_report = ito_sku_nodate_actual.merge(table_ito_sku_nodate, on=['fdc', 'sku'])

    # 3、保存报表
    Sku_report.to_csv(save_path + os.sep + 'Sku_report.csv')
