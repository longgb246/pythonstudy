#!/usr/bin/env python3
# coding:utf-8
__author__ = 'wangxin52'

'''
Table name: 1.app_ioa_iaa_dayerr_tmp（前置表，便于剔除异常值） 2.ap.app_ioa_iaa_dayerr 3.app.app_ioa_iaa_std
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
    end_dt = (datetime.date.today() - datetime.timedelta(num)).strftime(format)
    start_dt = (datetime.date.today() - datetime.timedelta(num + 60)).strftime(format)
    return start_dt, end_dt


def main():
    if (len(sys.argv) > 1):  # 传了系统参数
        end_dt = sys.argv[1]
    else:
        end_dt = get_format_yesterday()[1]

    sql = """
insert overwrite table app.app_ioa_iaa_dayerr_tmp partition (dt='""" + end_dt + """')
select
    t1.sku_id,
    t1.fdc_id,
    day_err3,
    day_err7,
    row_number() over (partition by t1.sku_id,t1.fdc_id order by day_err3) as rank3,
    row_number() over (partition by t1.sku_id,t1.fdc_id order by day_err7) as rank7,
    t2.cnt
from 
    (
        select
            dt,
            sku_id,
            fdc_id,
            case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
                else coalesce(sales_sum3,0)-forecast_daily_override_sales[0]-forecast_daily_override_sales[1]-forecast_daily_override_sales[2] end day_err3,
            case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
                else coalesce(sales_sum7,0)-forecast_daily_override_sales[0]-forecast_daily_override_sales[1]-forecast_daily_override_sales[2]-
                    forecast_daily_override_sales[3]-forecast_daily_override_sales[4]-forecast_daily_override_sales[5]-forecast_daily_override_sales[6]  end day_err7
        from
            app.app_ioa_iaa_stdpre
        where
            dp     = '""" + end_dt + """'
            and sku_status_cd=1
    ) t1
join
    (
        select
            sku_id,
            fdc_id,
            count(dt) cnt
        from
            app.app_ioa_iaa_stdpre
        where
            dp     = '""" + end_dt + """'
            and sku_status_cd=1
        GROUP BY
            sku_id,
            fdc_id
    )t2
on
    t1.sku_id=t2.sku_id and
    t1.fdc_id=t2.fdc_id;


insert overwrite table app.app_ioa_iaa_dayerr  partition(dp='""" + end_dt + """')
select
    case when t1.dt is not null then t1.dt else t2.dt end as dt,
    case when t1.sku_id is not null then t1.sku_id else t2.sku_id end as sku_id,
    case when t1.fdc_id is not null then t1.fdc_id else t2.fdc_id end as fdc_id,
    day_err3,
    day_err7
from
    (
        select
            dt,
            sku_id,
            fdc_id,
            day_err3
        from
        app.app_ioa_iaa_dayerr_tmp
        where
            rank3<= 0.95*cnt
            and dt='""" + end_dt + """'
    ) t1
full outer join
    (
        select
            dt,
            sku_id,
            fdc_id,
            day_err7
        from
            app.app_ioa_iaa_dayerr_tmp
        where
            rank7<= 0.95*cnt
            and dt='""" + end_dt + """'
    ) t2
on
    t1.dt=t2.dt and
    t1.sku_id=t2.sku_id and
    t1.fdc_id=t2.fdc_id;


insert overwrite table app.app_ioa_iaa_std partition(dt='""" + end_dt + """')
select
    t1.sku_id,
    t1.fdc_id,
    t1.std3,
    t1.std7,
    case when forecast_daily_override_sales is null then 0
        else forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+1.96*std3 end as lop,
    (8/7)*(forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+
        forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])+1.96*std3 as ti3,
    (8/7)*(forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+
        forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])+1.96*std7 as ti7
from
    (
        select
            dp,
            sku_id,
            fdc_id,
            stddev_pop(day_err3) std3,
            stddev_pop(day_err7) std7
        from
            app.app_ioa_iaa_dayerr
        where
            dp='""" + end_dt + """'
        group by
            dp,sku_id,fdc_id
    ) t1
left join
    (
        select
            dt,
            dc_id  as  fdc_id,
            sku_id,
            forecast_daily_override_sales
        from
            app.app_pf_forecast_result_fdc_di
        where
            dt = '""" + end_dt + """'
    ) t2
on
    t1.sku_id=t2.sku_id
    and t1.fdc_id=t2.fdc_id
    and t1.dp=t2.dt;
"""
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)


if __name__ == "__main__":
    main()
