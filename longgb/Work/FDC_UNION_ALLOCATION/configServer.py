# coding: utf-8
import platform
import os.path
import sys
import time
import numpy as np
data_dir        ='D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/'
output_dir      = 'D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/simulation_results/'
log_path         ='D:/Lgb/WorkFiles/FDC_UNION_ALLOCATION/log/log.txt'
data_file_name='sales_all_detail_20170324073521.csv'
rdc_inv_file_name='rdc_inv_big0_03_20170324140531.csv'
order_file_name='order_sample_070203_20170324143010.csv'
rdc_sale_file_name='rdc_Sale_data_070203_20170324143241.csv'
# date_range=[['2016-12-01','2016-12-14'],['2016-12-01','2016-12-21'],['2016-12-01','2016-12-28'],['2016-12-01','2017-01-30']]
date_range=[['2016-12-02','2016-12-15'],['2017-02-19','2017-03-20']]
date_end='2017-03-20'
date_start='2017-02-19'
vlt_val={'605':[1,2],'634':[2,3],'635':[3,4]}
vlt_prob={'605':[0.9944,0.0056],'634':[0.9944,0.0056],'635':[0.9944,0.0056]}