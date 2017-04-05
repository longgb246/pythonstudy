#-*- coding:utf-8 -*-
import os
import pandas as pd


path = r'D:\Lgb\data_sz\data'

file1_st = 'dev_inv_opt_simulation_data_seconde_pre_mid09_fdcall.csv'
file2_st = 'dev_inv_opt_simulation_order_fdcall.csv'
file3_st = 'dev_inv_opt_simulation_rdc_inv_data_fdcall.csv'
file4_st = 'dev_inv_opt_simulation_rdc_sale_data_fdcall_2.csv'

file1_name = 'fdc_sales_select_20170330124541.csv'
file2_name = 'order_select_20170330125041.csv'
file3_name = 'rdc_inv_select_20170330125151.csv'
file4_name = 'rdc_sales_select_20170330124551.csv'

file1_read = pd.read_table(path + os.sep + file1_st, header=None)
file2_read = pd.read_table(path + os.sep + file2_st, header=None)
file3_read = pd.read_table(path + os.sep + file3_st, header=None)
file4_read = pd.read_table(path + os.sep + file4_st, header=None)


# dev.dev_inv_opt_simulation_data_seconde_pre_mid09_fdcall
file1_read.columns = ['dt', 'fdcid', 'sku_id', 'forecast_daily_override_sales', 'total_sales', 'stock_qtty',
                      'safestock', 'maxstock', 'stock_qtty_real', 'std', 'white_flag']
# dev_inv_opt_simulation_order_fdcall
file2_read.columns = ['arrive_time', 'item_sku_id', 'arrive_quantity', 'int_org_num']
# dev_inv_opt_simulation_rdc_inv_data_fdcall
file3_read.columns = ['dt', 'delv_center_num', 'sku_id', 'stock_qtty']
# dev_inv_opt_simulation_rdc_sale_data_fdcall
file4_read.columns = ['sku_id', 'dc_id', 'order_date', 'total_sales', 'sales_as_gift', 'sales_for_free',
                      'big_order_sales', 'special_order_sales', 'order_qtty', 'dt']


file1_read.to_csv(path + os.sep + file1_name,sep='\t', index=False)
file2_read.to_csv(path + os.sep + file2_name,sep='\t', index=False)
file3_read.to_csv(path + os.sep + file3_name,sep='\t', index=False)
file4_read.to_csv(path + os.sep + file4_name,sep='\t', index=False)
