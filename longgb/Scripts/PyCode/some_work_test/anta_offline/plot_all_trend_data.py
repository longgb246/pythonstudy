# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/9/17
  Usage   : 
"""

# 源数据表：app.app_saas_sfs_model_input
# 线上数据表：app.app_saas_sfs_rst
# 预测数据表：app.app_lgb_test_bca_forecast_result_all_try

import sys
import os
import datetime
import argparse
import pprint

# 源数据表
sql_real = """
select 
    sale_date,
    sum(sale) as sale
from     
    app.app_saas_sfs_model_input
group by 
    sale_date  
"""

sql_pre = """
select 
    {select_str}
from
    {table}
where 
    tenant_id=28 and dt='{sfs_date}' {model_str}
"""

pre_len = 91
select_list = []
model_str = ''
for i in range(pre_len):
    select_list.append(
        "\n    sum(cast(split(substring(sale_list, 2, length(sale_list)-2),',')[{i}] as double)) as pre_sale_{i}".format(
            i=i))
select_str = ','.join(select_list)


def self_parse_args():
    yesterday = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y-%m-%d')
    config = {'table': 'app.app_lgb_test_bca_forecast_result_try', 'dt': yesterday}

    parser = argparse.ArgumentParser(description='get data detail')
    optional_prefix = '--'
    for k, v in config.items():
        parser.add_argument(optional_prefix + k, nargs='?', help='default : ' + v, default=v)

    config = parser.parse_args()
    config_dict = config.__dict__
    dt = config_dict.get('dt')

    pprint.pprint(config_dict)
    return config_dict, dt


def get_data(sale_sql='', pre_on_sql='', pre_test_sql='', dt=''):
    hive_1 = ''' hive -e "{hive_sql}" > sale_active.tsv '''.format(hive_sql=sale_sql)
    print('\n{0}\n'.format(hive_1))
    os.system(hive_1)

    hive_2 = ''' hive -e "{hive_sql}" > pred_on_{dt}.tsv '''.format(hive_sql=pre_on_sql.format(dt=dt), dt=dt)
    print('\n{0}\n'.format(hive_2))
    os.system(hive_2)

    hive_2 = ''' hive -e "{hive_sql}" > pred_test_{dt}.tsv '''.format(hive_sql=pre_test_sql.format(dt=dt), dt=dt)
    print('\n{0}\n'.format(hive_2))
    os.system(hive_2)

    try:
        os.system(''' /usr/bin/rm -rf plot_data.tar.gz ''')
    except:
        pass
    exec_str = ''' tar -zcvf plot_data.tar.gz sale_active.tsv pred_on_{dt}.tsv pred_test_{dt}.tsv '''. \
        format(dt=dt)
    os.system(exec_str)




if __name__ == '__main__':
    config_dict, dt = self_parse_args()
    # dt = '2018-09-16'
    pre_on_sql = sql_pre.format(select_str=select_str, table='app.app_saas_sfs_rst', sfs_date=dt, model_str=model_str)
    pre_test_sql = sql_pre.format(select_str=select_str, table='app.app_lgb_test_bca_forecast_result_all_try', sfs_date=dt, model_str=model_str)
    # print pre_test_sql
    get_data(sale_sql=sql_real, pre_on_sql=pre_on_sql, pre_test_sql=pre_test_sql, dt=dt)
