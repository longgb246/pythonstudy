# coding: utf-8
import platform
import os.path
import sys
import time
import numpy as np
data_dir        ='D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/'
output_dir      = 'D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/analysis_3_policy/Simulation_Results/simulation_results_select/'
log_path         ='D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/analysis_3_policy/Simulation_Results/log/log.txt'
data_file_name='fdc_sales_select_20170330124541.csv'
rdc_inv_file_name='rdc_inv_select_20170330125151.csv'
order_file_name='order_select_20170330125041.csv'
rdc_sale_file_name='rdc_sales_select_20170330124551.csv'
# date_range=[['2016-12-01','2016-12-14'],['2016-12-01','2016-12-21'],['2016-12-01','2016-12-28'],['2016-12-01','2017-01-30']]
date_range=[['2016-12-02','2016-12-15'],['2017-02-19','2017-03-20']]
date_end='2017-03-20'
date_start='2017-02-19'
vlt_val={605:[1,2],633:[2,3],634:[2,3]}
vlt_prob={605:[0.9944,0.0056],633:[0.9944,0.0056],634:[0.9944,0.0056]}