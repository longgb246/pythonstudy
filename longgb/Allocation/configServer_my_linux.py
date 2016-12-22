#-*- coding:utf-8 -*-
import os

# ===============================================================
# =                         （1）系统配置                        =
# ===============================================================

data_path = r'/home/cmo_ipc/Allocation_shell/datasets'
data_path_total = r'/home/cmo_ipc/Allocation_shell/datasets/data_total3'

# 1、读取路径
sku_data_path = data_path_total + os.sep + 'total_sku' + os.sep + '2016-10-01.pkl'
fdc_data_path = data_path_total + os.sep + 'fdc_data.pkl'
order_data_path = data_path_total + os.sep + 'order_data.pkl'
sale_data_path = data_path_total + os.sep + 'total_sale' + os.sep + '2016-10-01.pkl'
sale_data_path_batch = data_path_total + os.sep + 'total_sale'
fdc_initialization_inv = data_path_total + os.sep + 'total_fdcinv' + os.sep + '2016-10-01.pkl'

# 2、储存路径
save_data_path = r'/home/cmo_ipc/Allocation_shell/longgb/allocation_test_three_day/'
log_path = r'/home/cmo_ipc/Allocation_shell/longgb/allocation_test_three_day/allocation.log'


# sku_data_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/total_sku/2016-10-01.pkl'
# fdc_data_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/fdc_data.pkl'
# order_data_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/order_data.pkl'
# sale_data_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/total_sale/2016-10-01.pkl'
# fdc_initialization_inv='/home/cmo_ipc/Allocation_shell/datasets/data_total3/total_fdcinv/2016-10-01.pkl'
# sale_data_path_batch='/home/cmo_ipc/Allocation_shell/datasets/data_total3/total_sale/'
# save_data_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/'#数据集存储路径
# log_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/allocation.log'#日志路口径


