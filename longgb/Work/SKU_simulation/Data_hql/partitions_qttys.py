#-*- coding:utf-8 -*-
from string import Template
import os
import pandas as pd

def getDateRange(start_date, end_date, freq='D'):
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date, freq=freq).values)
    return date_range


# 1、插入函数
dev_tmp_lgb_allocation_qttys = '''
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert OVERWRITE table dev.dev_tmp_lgb_allocation_qttys  partition(date_s)
select
    case when A.fdcid is not null then A.fdcid else B.fdc_id end as fdc_id,
    case when A.wid is not null then A.wid else B.sku_id end as sku_id,
    case when A.wid is not null then 1 else 0 end as is_whitelist,
    B.rdc_id,
    B.plan_num_auto as plan_allocation_qtty,
    B.delivered_num_auto as actual_allocation_qtty,
    "$this_date" as date_s
from
    (
    select
        t2.wid,
        t2.fdcid
    from
        dim.dim_dc_info t1
    join
        fdm.fdm_fdc_whitelist_chain t2
    on
        t1.dc_id = t2.fdcid
    where
        t2.start_date <= "$this_date" and
        t2.end_date > "$this_date" and
        to_date(t2.create_time) <= "$this_date" and
        to_date(t2.modify_time) <= "$this_date" and
        t2.yn = 1 and
        t1.dc_type = 1) A
full outer join
    (
    select
        to_date(ck.create_date) as dt,
        co.art_no as sku_id,
        ck.org_to as fdc_id,
        ck.org_from as rdc_id,
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then plan_num else 0 end) as plan_num_auto,                   -- 计划调拨量
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then delivered_num else 0 end) as delivered_num_auto          -- 实际调拨量
    from
        (select * from dim.dim_dc_info where dc_type = 1) di                    -- 配送中心所属关系， 取 dc_type = 1， 1-FDC。
    join
        fdm.fdm_newdeploy_chuku_chain ck                                        -- 内配计划出库表（内配单）
    on
        di.dc_id = ck.org_to
    join
        fdm.fdm_newdeploy_chuorders_chain co                                    -- 未知表，看看。
    on
        ck.id = co.chuku_id
    where
        co.dp = "ACTIVE" and
        ck.dp = "ACTIVE" and
        ck.yn in (1, 3, 5) and                                                  -- 1---正常，3---删除处理中， 5---删除失败
        -- ck.org_from in (3, 4, 5, 6, 9, 10, 316, 682) and                     -- 配出机构
        ck.org_from = 4 and                                                     -- 只取出 RDC 为 4 的
        ck.org_to = 605 and                                                     -- 只取出 FDC 为 605 的
        to_date(ck.create_date) = "$this_date"
    group by
        to_date(ck.create_date),
        co.art_no,
        ck.org_to,
        ck.org_from) B
on
    A.wid = B.sku_id and
    A.fdcid = B.fdc_id;
'''
dev_tmp_lgb_allocation_qttys = Template(dev_tmp_lgb_allocation_qttys)

dev_tmp_lgb_combine_allocation_Forecast = '''
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert OVERWRITE table dev.dev_tmp_lgb_combine_allocation_Forecast  partition(date_s)
select
    a.rdc_id,
    a.fdc_id,
    a.sku_id,
    a.is_whitelist,
    a.plan_allocation_qtty,
    a.actual_allocation_qtty,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean,
    d.sale,
    d.sale_all,
    d.stock_qtty,
    a.date_s
from
    (
        select
            a.rdc_id,
            a.fdc_id,
            a.sku_id,
            a.is_whitelist,
            a.plan_allocation_qtty,
            a.actual_allocation_qtty,
            a.date_s
        from
            dev.dev_tmp_lgb_allocation_qttys a
        where
            a.fdc_id = 605  AND
            a.rdc_id = 4    AND
            a.date_s = "$this_date"
    ) a
left join
    (
        select
            b.sku_id,
            b.forecast_daily_override_sales,
            b.forecast_sales_mean,
            b.sale,
            b.sale_all,
            c.stock_qtty
        from
            (
                select
                    m.sku_id,
                    m.forecast_daily_override_sales,
                    m.forecast_sales_mean,
                    m.sale,
                    n.sale_all
                from
                    (
                        -- 取20号的数据
                        select
                            wid as sku_id,
                            forecast_daily_override_sales,
                            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean,
                            total_sales  as sale
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605 and
                            dt = "$this_date"
                    )   m
                left join
                    (
                        -- 取平均的销量
                        select
                            wid as sku_id,
                            avg(total_sales)   as sale_all
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605     and
                            dt >= "$avg_date"
                        group by
                            wid
                    )   n
                on
                    m.sku_id = n.sku_id
            )  b
        left join
            (
                -- 取inv的前一天数据
                select
                    wid as sku_id,
                    stock_qtty
                from
                    dev.dev_tmp_lgb_saleForecast
                where
                    fdcid = 605 and
                    dt = "$last_date"
            )  c
        on
            b.sku_id = c.sku_id
    ) d
on
    a.sku_id = d.sku_id;
'''
dev_tmp_lgb_combine_allocation_Forecast = Template(dev_tmp_lgb_combine_allocation_Forecast)

# 2、检查函数
dev_tmp_lgb_allocation_qttys_check = '''select
    *
from
    dev.dev_tmp_lgb_allocation_qttys
where
    date_s = "$this_date"   and
    rdc_id = 4 and
    fdc_id = 605
limit
    50;
'''
dev_tmp_lgb_allocation_qttys_check = Template(dev_tmp_lgb_allocation_qttys_check)

dev_tmp_lgb_combine_allocation_Forecast_check = '''select
    *
from
    dev.dev_tmp_lgb_combine_allocation_Forecast
where
    date_s = "$this_date"   and
    rdc_id = 4 and
    fdc_id = 605
limit
    50;
'''
dev_tmp_lgb_combine_allocation_Forecast_check =Template(dev_tmp_lgb_combine_allocation_Forecast_check)


start_date = '2016-12-03'
end_date = '2016-12-31'


if __name__ == '__main__':
    date_range = getDateRange(start_date, end_date)
    for date_s in date_range:
        print "======================================================"
        print "                      {0}".format(date_s)
        print "======================================================"
        if date_s == '2016-12-20':
            pass
        else:
            os.system("hive -e '{0}';".format(dev_tmp_lgb_allocation_qttys.substitute(this_date=date_s)))
            # os.system('echo "======================================================"  >>  dev_tmp_lgb_allocation_qttys.log')
            # os.system('echo "======================= {0} ==================="  >>  dev_tmp_lgb_allocation_qttys.log'.format(date_s))
            # os.system('echo "======================================================  >>  dev_tmp_lgb_allocation_qttys.log"')
            # os.system("hive -e '{0}'    >>  dev_tmp_lgb_allocation_qttys.log;".format(dev_tmp_lgb_allocation_qttys_check.substitute(this_date=date_s)))
            # exit()
    pass
