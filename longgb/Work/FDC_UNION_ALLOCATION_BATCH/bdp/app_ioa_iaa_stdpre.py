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
    start_dt = (datetime.date.today() - datetime.timedelta(num+71)).strftime(format)
    drop_dt =  (datetime.date.today() - datetime.timedelta(num+15)).strftime(format)
    byesterday = (datetime.date.today() - datetime.timedelta(num+1)).strftime(format)
    return start_dt,end_dt,byesterday,drop_dt



def main():

    if(len(sys.argv)>1): # 传了系统参数
        end_dt = sys.argv[1]
        start_dt = (datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d') - datetime.timedelta(71)).strftime( '%Y-%m-%d')
        drop_dt = (datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d') - datetime.timedelta(15)).strftime( '%Y-%m-%d')
        byesterday = (datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d') - datetime.timedelta(1)).strftime( '%Y-%m-%d')
    else:
        end_dt = get_format_yesterday()[1]
        start_dt = get_format_yesterday()[0]
        byesterday = get_format_yesterday()[2]
        drop_dt = get_format_yesterday()[3]
        # """
    # drop table if exists app.app_ioa_iaa_skudc;
    #     create table app.app_ioa_iaa_skudc(
    #     sku_id string,
    #     fdc_id string,
    #     sku_status_cd string)
    # partitioned by(dt string)
    # stored as TEXTFILE;
    #     drop table if exists app.app_ioa_iaa_stdpre;
    #     create table app.app_ioa_iaa_stdpre(
    #     sku_id string comment '商品编号',
    #     fdc_id string comment 'FDC编号',
    #     sku_status_cd string comment '上下柜状态，1,0',
    #     forecast_daily_override_sales  array<double> comment 'FDC预测销量',
    #     sales_sum1 string comment '每日销量',
    #     sales_sum2 string comment '2天销量和',
    #     sales_sum3 string comment '3天销量和',
    #     sales_sum4 string comment '4天销量和',
    #     sales_sum5 string comment '5天销量和',
    #     sales_sum6 string comment '6天销量和',
    #     sales_sum7 string comment '7天销量和') comment '残差标准差计算前置表'
    # partitioned by(dt string)
    # stored as TEXTFILE;
    # """

    sql = """
    insert overwrite table app.app_ioa_iaa_skudc partition(dt='"""+byesterday+"""')
    select
        t1.sku_id,
        t2.fdc_id,
        t1.sku_status_cd
    from
        (
            select
                s.sku_id,
                s.sku_status_cd
            from
                (
                    select
                        r1.sku_id,
                        r1.sku_status_cd,
                        r2.label
                    from
                        (
                            select
                                item_sku_id as sku_id,
                                sku_status_cd
                            from
                                gdm.gdm_m03_item_sku_da
                            where
                                dt = '"""+byesterday+"""'
                                and    data_type = '1' -- 1，自营；
                                and    sku_valid_flag = 1
                                and    vender_direct_delv_flag in ('0','NULL','')
                        ) r1
                    left join
                        (--取出厂直和售完即止
                            select
                                distinct product_sku_id as sku_id,
                                1 as label
                            from
                                fdm.fdm_forest_productext_da
                            where
                                dt = '"""+byesterday+"""'
                                and ((dtype      = 'factoryShip' and dvalue = '1')
                                or (dtype      = 'SaleNo' and dvalue = '1'))
                         ) r2
                    on
                        r1.sku_id=r2.sku_id
                ) s
            where
                label is null
        ) t1
    cross join
        (--FDC仓编号
            select
                 dc_id as fdc_id
            from
                 dim.dim_dc_info
            where
                dc_type = 1
        ) t2;


    insert overwrite table app.app_ioa_iaa_stdpre partition(dt='"""+byesterday+"""')
    select
        sku_id,
        fdc_id,
        sku_status_cd,
        forecast_daily_override_sales,
        sales_sum1,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) as sales_sum2,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) as sales_sum3,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING) as sales_sum4,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) as sales_sum5,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING) as sales_sum6,
        sum(sales_sum1) over(partition  by fdc_id,sku_id order by dt  ROWS BETWEEN CURRENT ROW AND 6 FOLLOWING) as sales_sum7
    from
        (
            select
                t1.dt,
                t1.sku_id,
                t1.fdc_id,
                case when t2.sku_status_cd is not null then t2.sku_status_cd
                when t2.sku_status_cd is null and t1.sku_status_cd ='3001' then 1
                else 0 end as sku_status_cd,
                t3.forecast_daily_override_sales,
                coalesce(t4.total_sales,0) as sales_sum1
            from
                (
                    select
                        dt,
                        sku_id,
                        fdc_id,
                        sku_status_cd
                    from
                        app.app_ioa_iaa_skudc
                    where
                        dt='"""+byesterday+"""'
                ) t1
            left join
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
                        dt = '"""+byesterday+"""'
                    group by
                        dt,
                        dim_delv_center_num,
                        sku_id
                ) t2
            on
                t1.sku_id=t2.sku_id
                and t2.fdc_id=t1.fdc_id
            left join
                (--销量预测
                    select
                        dt,
                        dc_id  as  fdc_id,
                        sku_id,
                        forecast_daily_override_sales
                    from
                        app.app_pf_forecast_result_fdc_di
                    where
                        dt = '"""+byesterday+"""'
                        and dc_type='1'
                ) t3
            on
                t1.sku_id=t3.sku_id
                and t1.fdc_id=t3.fdc_id
            left join
                (--每日销量。已验证无重复数据
                    select
                        dt,
                        dc_id  as  fdc_id,
                        sku_id,
                        total_sales
                    from
                        app.app_sfs_sales_dc
                    where
                        dt = '"""+byesterday+"""'
                ) t4
            on
                t1.sku_id=t4.sku_id and
                t1.fdc_id=t4.fdc_id
        ) s;
    """
    print(sql)
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=sql)

if __name__ == "__main__":
    main()