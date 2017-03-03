#-*- coding:utf-8 -*-
import os
import pandas as pd


read_path = r'D:\Lgb\data_sz'
save_path = r'D:\Lgb\data_local'


def dev_fdc_daily_sales_inv_sku():
    sales_path = read_path + os.sep + 'dev.dev_fdc_daily_sales_inv_sku'
    sales_data = pd.read_table(sales_path, header=None)
    sales_data.columns = ['dc_id', 'date_s', 'item_sku_id', 'total_sales']
    sales_data.to_csv(save_path + os.sep + 'dev_fdc_daily_sales_inv_sku.csv', index=False)


def dev_fdc_daily_total_inv_sku():
    inv_path = read_path + os.sep + r'dev_fdc_daily_total_inv_sku\dev.dev_fdc_daily_total_inv_sku'
    inv_data = pd.read_table(inv_path, header=None)
    inv_data.columns = ['dc_id', 'date_s', 'sku_id', 'total_inv']
    inv_data.to_csv(save_path + os.sep + 'dev_fdc_daily_total_inv_sku.csv', index=False)


def hive_02_select_from_big():
    hive_02_select_from_big = pd.read_table(read_path + os.sep + 'hive_02_select_from_big.out', header=None)
    hive_02_select_from_big.columns = ['dc_id', 'sku_id', 'date_s', 'inv_actual', 'inv_sim', 'white_flag', 'sales_mean', 'safe_qtty', 'max_qtty', 'sale_ord_id', 'parent_sale_ord_id', 'sale_ord_white_flag', 'sale_qtty_detail', 'sale_qtty_actual', 'sale_qtty_sim']
    hive_02_select_from_big.to_csv(save_path + os.sep + 'hive_02_select_from_big.csv', index=False)


def hive_02_select_from_big_sim_arr():
    hive_02_select_from_big_sim_arr = pd.read_table(read_path + os.sep + 'hive_02_select_from_big_sim_arr.out', header=None)
    hive_02_select_from_big_sim_arr.columns = ['dc_id','sku_id','date_s','white_flag','sales_mean','safe_qtty','max_qtty','inv_rdc','inv_actual','inv_sim','open_po_fdc_actual','open_po_fdc_sim','cons_open_po_sim','arrive_quantity_sim','allocation_sim','sale_ord_id','parent_sale_ord_id','sale_ord_white_flag','sale_qtty_detail','sale_qtty_actual','sale_qtty_sim']
    hive_02_select_from_big_sim_arr.to_csv(save_path + os.sep + 'hive_02_select_from_big_sim_arr.csv', index=False)


if __name__ == '__main__':
    # dev_fdc_daily_sales_inv_sku()
    # dev_fdc_daily_total_inv_sku()
    # hive_02_select_from_big()
    hive_02_select_from_big_sim_arr()
    pass

