#!/usr/bin/env python3
# coding:utf-8

__author__ = 'guoxubo'

import sys
import os
import datetime
import time

sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask

def get_format_yesterday(format = '%Y-%m-%d'):
    """
    获取昨天日期（字符串）默认'%Y-%m-%d'，
    format =‘%d' 获取昨天是本月中第几天
    """
    yesterday = (datetime.date.today() + datetime.timedelta(-1)).strftime(format)
    return yesterday



def main():

    location_prefix = '/user/cmo_ipc_simulate/ipc_sim/history'

    if(len(sys.argv)>1): # 传了系统参数
        yesterday = sys.argv[1]
    else:
        yesterday = get_format_yesterday()


    yesterday_monthDay =  datetime.datetime.strptime(yesterday, '%Y-%m-%d').strftime('%d')  # 当月几号

    sql = """

        set mapred.job.priority=VERY_HIGH;
        set hive.exec.compress.output=true;
        set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec ;
        set hive.exec.dynamic.partition.mode=nonstrict;
        set hive.exec.dynamic.partition=true;
        SET hive.exec.max.dynamic.partitions=100000;
        SET hive.exec.max.dynamic.partitions.pernode=100000;

        create table if not exists app.app_dashboard_agg_credit_day(
            rowkey string  comment 'hbase rowkey'
            ,amount_payable string comment '应付金额'
            ,amount_payable_chain_indx string comment '应付金额环比'
            ,average_inventory_amount string comment '平均库存金额'
            ,average_inventory_amount_chain_indx string comment '平均库存金额环比'
            ,payable_turnover_ratio string comment '应付周转'
            ,payable_turnover_ratio_chain_indx string comment '应付周转环比'
            ,inventory_turnover_ratio string comment '存货周转'
            ,inventory_turnover_chain_indx string comment '存货周转环比'
            ,credit_days string comment '信用天数'
            ,credit_days_chain_indx string comment '信用天数环比'
        )
        partitioned by (type string,dt string);

        use app;
        alter table app_dashboard_agg_credit_day drop if exists partition(type='d',dt='""" + yesterday + """');
        --- 直接存放路径
        INSERT OVERWRITE DIRECTORY '""" + location_prefix + """no_book_logical_batch/""" + yesterday + """'
        select * from table;
        ---- 存放到某张表的 路径下
        INSERT OVERWRITE table app.app_oih3_vreturn_priority_chain PARTITION(dt = '""" + yesterday + """')
        select * from table;
        insert overwrite table app_dashboard_agg_credit_day partition(type='d',dt='""" + yesterday + """')
        select * from table;
    """
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)

if __name__ == "__main__":
    main()