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
insert overwrite table app.app_ioa_iaa_dayerr_tmp partition (dp='""" + end_dt + """')
select
    dt,
    sku_id,
    fdc_id,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum1,0)-forecast_daily_override_sales[0]  end day_err1,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum2,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]  end day_err2,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum3,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]-
    forecast_daily_override_sales[2] end day_err3,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum4,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]-
    forecast_daily_override_sales[2]-
    forecast_daily_override_sales[3]  end day_err4,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum5,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]-
    forecast_daily_override_sales[2]-
    forecast_daily_override_sales[3]-
    forecast_daily_override_sales[4]  end day_err5,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum6,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]-
    forecast_daily_override_sales[2]-
    forecast_daily_override_sales[3]-
    forecast_daily_override_sales[4]-
    forecast_daily_override_sales[5]  end day_err6,
    case when forecast_daily_override_sales is null then coalesce(sales_sum3,0)
    else
    coalesce(sales_sum7,0)-forecast_daily_override_sales[0]-
    forecast_daily_override_sales[1]-
    forecast_daily_override_sales[2]-
    forecast_daily_override_sales[3]-
    forecast_daily_override_sales[4]-
    forecast_daily_override_sales[5]-
    forecast_daily_override_sales[6]  end day_err7
from
    app.app_ioa_iaa_stdpre
where
    dp     = '""" + end_dt + """'
    and sku_status_cd=1;

insert overwrite table app.app_ioa_iaa_dayerr partition(dp='""" + end_dt + """')
select
    t1.dt,
    t1.sku_id,
    t1.fdc_id,
    case when day_err1>=percent1 then NULL else day_err1 end as day_err1,
    case when day_err2>=percent2 then NULL else day_err2 end as day_err2,
    case when day_err3>=percent3 then NULL else day_err3 end as day_err3,
    case when day_err4>=percent4 then NULL else day_err4 end as day_err4,
    case when day_err5>=percent5 then NULL else day_err5 end as day_err5,
    case when day_err6>=percent6 then NULL else day_err6 end as day_err6,
    case when day_err7>=percent7 then NULL else day_err7 end as day_err7
from
(select
    dt,
    sku_id,
    fdc_id,
    day_err1,
    day_err2,
    day_err3,
    day_err4,
    day_err5,
    day_err6,
    day_err7
from
    app.app_ioa_iaa_dayerr_tmp
where
    dp='""" + end_dt + """') t1
join
(select
    sku_id,
    fdc_id,
/*    percentile_approx(cast(day_err1 as double),array(0.85,0.90,0.95),9999) percent1,
    percentile_approx(cast(day_err2 as double),array(0.85,0.90,0.95),9999) percent2,
    percentile_approx(cast(day_err3 as double),array(0.85,0.90,0.95),9999) percent3,
    percentile_approx(cast(day_err4 as double),array(0.85,0.90,0.95),9999) percent4,
    percentile_approx(cast(day_err5 as double),array(0.85,0.90,0.95),9999) percent5,
    percentile_approx(cast(day_err6 as double),array(0.85,0.90,0.95),9999) percent6,
    percentile_approx(cast(day_err7 as double),array(0.85,0.90,0.95),9999) percent7,*/
    percentile_approx(cast(day_err1 as double),0.95,9999) percent1,
    percentile_approx(cast(day_err2 as double),0.95,9999) percent2,
    percentile_approx(cast(day_err3 as double),0.95,9999) percent3,
    percentile_approx(cast(day_err4 as double),0.95,9999) percent4,
    percentile_approx(cast(day_err5 as double),0.95,9999) percent5,
    percentile_approx(cast(day_err6 as double),0.95,9999) percent6,
    percentile_approx(cast(day_err7 as double),0.95,9999) percent7,
from
    app.app_ioa_iaa_dayerr_tmp
where
    dp='""" + end_dt + """'
group by
    sku_id,fdc_id) t2
on t1.sku_id=t2.sku_id and t1.fdc_id =t2.fdc_id;

insert overwrite table app.app_ioa_iaa_std partition(dt='""" + end_dt + """')
    select
        sku_id,
        fdc_id,
        stddev_pop(day_err1) std1,
        stddev_pop(day_err2) std2,
        stddev_pop(day_err3) std3,
        stddev_pop(day_err4) std4,
        stddev_pop(day_err5) std5,
        stddev_pop(day_err6) std6,
        stddev_pop(day_err7) std7
    from
        app.app_ioa_iaa_dayerr
    where
        dp='""" + end_dt + """'
    group by
        dp,sku_id,fdc_id;
"""
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)


if __name__ == "__main__":
    main()
