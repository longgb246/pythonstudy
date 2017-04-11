#!/usr/bin/env python3
# coding:utf-8
__author__ = 'wangxin52'

'''
Table name: 1.ap.app_ioa_iaa_dayerr2.app.app_ioa_iaa_std
Description: FDC三日，七日标准差
Author: wangxin52@jd.com
'''
import sys
import os
import datetime
import time

sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask


def get_format_yesterday(num=1, format='%Y-%m-%d'):
    """
    获取昨天日期（字符串）默认'%Y-%m-%d'，
    format =‘%d' 获取昨天是本月中第几天
    """
    into_dt = (datetime.date.today() - datetime.timedelta(num + 1)).strftime(format)
    end_dt = (datetime.date.today() - datetime.timedelta(num)).strftime(format)
    return end_dt, into_dt


def main():
    if (len(sys.argv) > 1):  # 传了系统参数
        into_dt = sys.argv[1]
        end_dt = (datetime.datetime.strptime(sys.argv[1], '%Y-%m-%d') + datetime.timedelta(1)).strftime('%Y-%m-%d')
    else:
        into_dt = get_format_yesterday()[1]
        end_dt = get_format_yesterday()[0]

    sql = """
insert overwrite table app.app_ioa_iaa_whitelist partition(dt='""" + end_dt + """')
select
        t2.wid sku_id,
        t2.fdcid fdc_id
    from
        dim.dim_dc_info t1
    join
        fdm.fdm_fdc_whitelist_chain t2
    on
        t1.dc_id = t2.fdcid
    where
        t2.start_date <= '""" + into_dt + """' and
        t2.end_date > '""" + into_dt + """' and
        t2.yn = 1 and
        t1.dc_type = 1;
"""
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)


if __name__ == "__main__":
    main()
