# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@jd.com
  Date    : 2018/8/15
  Usage   : 
"""

import os
import datetime
import argparse
import pprint

get_sale_sql = '''
    set hive.cli.print.header=true;
    select
        sku_code,
        store_id,
        category_code_1 as cate1,
        category_code_2 as cate2,
        brand_code_1 as brand,
        available_qty as qty,
        sale,
        sale_date as dt
    from
        app.app_saas_sfs_model_input
    where
        tenant_id = 28
        and dt = 'ACTIVE'    
    '''

get_pre_sql = '''
    set hive.cli.print.header=true;
    select
        split(sku_id, '_')[0] as sku_code,
        split(sku_id, '_')[1] as store_id,
        sale_type,
        sale_list
    from
        app.app_lgb_test_bca_forecast_result_all_try
    where
        tenant_id = 28
        and dt = '{dt}'        
    '''


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


def get_data(sale_sql='', pre_sql='', dt=''):
    hive_1 = ''' hive -e "{hive_sql}" > sale_active.tsv '''.format(hive_sql=sale_sql)
    print('\n{0}\n'.format(hive_1))
    os.system(hive_1)

    hive_2 = ''' hive -e "{hive_sql}" > pred_{dt}.tsv '''.format(hive_sql=pre_sql.format(dt=dt), dt=dt)
    print('\n{0}\n'.format(hive_2))
    os.system(hive_2)

    try:
        os.system(''' /usr/bin/rm -rf plot_data.tar.gz ''')
    except:
        pass
    exec_str = ''' tar -zcvf plot_data.tar.gz sale_active.tsv pred_{dt}.tsv '''.format(dt=dt)
    os.system(exec_str)


def dict_update(org, update, mode='force'):
    """
    Recursive update the dict.
    :param org: org dict
    :param update: update dict
    :param mode: 'force' - if the key is same, though value type is not same update the value
    """
    for k, v in update.items():
        tmp_v = org.get(k)
        if tmp_v is None:
            org[k] = update[k]
        elif isinstance(tmp_v, dict) and isinstance(update[k], dict):
            dict_update(tmp_v, update[k])
        elif mode == 'force':
            org[k] = update[k]
        elif type(org[k]) == type(update[k]):
            org[k] = update[k]
    return org


if __name__ == '__main__':
    config_dict, dt = self_parse_args()
    get_data(sale_sql=get_sale_sql, pre_sql=get_pre_sql, dt=dt)
