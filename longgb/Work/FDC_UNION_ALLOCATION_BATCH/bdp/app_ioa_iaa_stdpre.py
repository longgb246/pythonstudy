#!/usr/bin/env python3
# coding:utf-8
__author__ = 'wangxin52'

'''
Table name: app.app_ioa_iaa_stdpre
Description: FDC标准差前置宽表
Author: wangxin52@jd.com
'''
import sys
import os
import datetime
import time

sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask

def get_format_yesterday(num=1,format = '%Y-%m-%d'):
    """
    获取昨天日期（字符串）默认'%Y-%m-%d'，
    format =‘%d' 获取昨天是本月中第几天
    """
    end_dt = (datetime.date.today() - datetime.timedelta(num)).strftime(format)
    start_dt = (datetime.date.today() - datetime.timedelta(num+60)).strftime(format)
    return start_dt,end_dt



def main():

    if(len(sys.argv)>1): # 传了系统参数
        end_dt = sys.argv[1]
        start_dt = (datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d') - datetime.timedelta(60)).strftime( '%Y-%m-%d')
    else:
        end_dt = get_format_yesterday()[1]
        start_dt = get_format_yesterday()[0]

        # """
    #     drop table if exists app.app_ioa_iaa_stdpre;
    #     create table app.app_ioa_iaa_stdpre(
    #     sku_id string comment '商品编号',
    #     fdc_id string comment 'FDC编号',
    #     dt string comment '日期',
    #     sku_status_cd string comment '上下柜状态，1,0',
    #     forecast_daily_override_sales  array<double> comment 'FDC预测销量',
    #     total_sales string comment '每日销量',
    # sales_sum3 string comment '3天销量和',
    # sales_sum7 string comment '7天销量和'
    # )partitioned by(dp string)
    # stored as TEXTFILE;
    # """

    sql = """
insert overwrite table app.app_ioa_iaa_stdpre partition(dp='"""+end_dt+"""')
select
	t1.sku_id,
	t1.fdc_id,
	t2.dt,
	t2.sku_status_cd,
	t3.forecast_daily_override_sales,
	t4.total_sales,
	sum(coalesce(t4.total_sales,0))over(partition  by t4.sku_id order by t4.dt  ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) as sales_sum3,
	sum(coalesce(t4.total_sales,0))over(partition  by t4.sku_id order by t4.dt  ROWS BETWEEN CURRENT ROW AND 6 FOLLOWING) as sales_sum7
from
	(--每日白名单
	select 
		sku_id,fdc_id
	from 
		app.app_ioa_iaa_whitelist 
	where dt='"""+end_dt+"""') t1
join
	(--上下柜
	select
	    -- 它有重复记录，可能存在 同一个dt、fdc_id、sku_id，但是库存、sku_status_cd不相同的情况
	    dt,
	    dim_delv_center_num  as fdc_id,
	    sku_id,
	    max(sku_status_cd)  as  sku_status_cd
	from
	    app.app_sfs_vendibility
	where
	    dt >= '"""+start_dt+"""'
	    and dt <= '"""+end_dt+"""'
	group by
	    dt,
	    dim_delv_center_num,
	    sku_id) t2
on t1.sku_id=t2.sku_id and t1.fdc_id=t2.fdc_id
join
	(--销量预测（60天）
	select
	    dt,
	    dc_id  as  fdc_id,
	    sku_id,
	    forecast_daily_override_sales
	from
    	app.app_pf_forecast_result_fdc_di
	where
	    dt >= '"""+start_dt+"""'
	    and dt <= '"""+end_dt+"""') t3
on t1.sku_id=t3.sku_id and t1.fdc_id=t3.fdc_id and t2.dt=t3.dt
join
	(--每日销量，取60天。已验证无重复数据
	select
	    dt,
	    dc_id  as  fdc_id,
	    sku_id,
	    total_sales
	from
	    app.app_sfs_sales_dc
	where
	    dt >= '"""+start_dt+"""'
	    and dt <= '"""+end_dt+"""') t4
on t3.sku_id=t4.sku_id and t3.fdc_id=t4.fdc_id and t3.dt=t4.dt;
"""
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)

if __name__ == "__main__":
    main()