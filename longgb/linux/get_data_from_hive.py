#-*- coding:utf-8 -*-
import os
from string import Template
import time

# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630/628/658 316

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))


def pyhadoop(table_str, log_str, dir_name):
    os.system('echo "{0}" >> {1}_log.log 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1}_log.log 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1}_log.log 2>&1;'.format('*'*50, log_str))
    os.system('hadoop fs -get /user/cmo_ipc/dev.db/{0}  /home/cmo_ipc/Allocation_shell/datasets/{2} > {1}_log.log 2>&1;'.format(table_str, log_str, dir_name))


# table_names = ['dev_allocation_sku_data','dev_allocation_fdc_data','dev_allocation_order_data','dev_allocation_sale_data']
table_names = ['dev_allocation_fdc_data','dev_allocation_order_data','dev_allocation_sale_data']
# dir_names = ['sku_datasets','fdc_datasets','order_datasets','sale_datasets']
dir_names = ['fdc_datasets','order_datasets','sale_datasets']

try:
    for i, table_name in enumerate(table_names):
        print 'Download data: {0}...'.format(table_name)
        pyhadoop(table_name, table_name, dir_names[i])
    os.system('echo "Read Data Success!" > finish.log 2>&1;')
except:
    table_names[0] += '_log.log'
    table_str = reduce(lambda x,y: x + ' , ' + y + '_log.log',table_names)
    os.system('''echo "Read Data False! Please read {0} ." > finish.log 2>&1;'''.format(table_str))

