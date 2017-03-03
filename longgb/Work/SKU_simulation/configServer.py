# coding: utf-8
import platform
import os.path
import sys
import time
import numpy as np
data_dir        ='E:/Allocation_data/'
output_dir      = 'E:/simulation_results/'
log_dir         ='E:/Allocation_data/log/'
data_file_name='sku_allocation_sample2.csv'
# date_range=[['2016-12-01','2016-12-14'],['2016-12-01','2016-12-21'],['2016-12-01','2016-12-28'],['2016-12-01','2017-01-30']]
date_range=[['2016-12-01','2016-12-15'],['2016-12-16','2016-12-30']]
date_end='2017-01-30'
date_start='2016-12-01'
vlt_val=np.array([1,2])
vlt_prob=np.array([0.9944,0.0056])