#-*- coding:utf-8 -*-
'''
该脚本有问题，在看看
问题为：hive中别随便使用单引号‘，它有可能认为是一列。
只取出 RDC : 4 - 成都, FDC : 605  - 重庆
建表2张：   dev.dev_tmp_lgb_allocation_qttys    调拨量数据
            dev.dev_tmp_lgb_saleForecast        销量预测数据
'''
from string import Template
import os
import pandas as pd


def getDate(start_date, end_date):
    return map(lambda x: str(x)[:10], list(pd.date_range(start_date, end_date)))


# 调拨量查询
hql_allocationQttys_create = '''
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
drop table if exists dev.dev_tmp_lgb_allocation_qttys;
CREATE TABLE IF Not EXISTS dev.dev_tmp_lgb_allocation_qttys
	(
		fdc_id	 string,
		sku_id	 string,
		is_whitelist	int,
		rdc_id	string,
		plan_allocation_qtty	double,
		actual_allocation_qtty	double)
		PARTITIONED by (date_s  string);
'''


hql_allocationQttys = '''
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert OVERWRITE table dev.dev_tmp_lgb_allocation_qttys partition(date_s)
select
    case when A.fdcid is not null then A.fdcid else B.fdc_id end as fdc_id,
    case when A.wid is not null then A.wid else B.sku_id end as sku_id,
    case when A.wid is not null then 1 else 0 end as is_whitelist,
    B.rdc_id,
    B.plan_num_auto as plan_allocation_qtty,
    B.delivered_num_auto as actual_allocation_qtty,
    '${dt}' as date_s
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
        t2.start_date <= '${dt}' and
        t2.end_date > '${dt}' and
        to_date(t2.create_time) <= '${dt}' and
        to_date(t2.modify_time) <= '${dt}' and
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
        to_date(ck.create_date) = '${dt}'
    group by
        to_date(ck.create_date),
        co.art_no,
        ck.org_to,
        ck.org_from) B
on
    A.wid = B.sku_id and
    A.fdcid = B.fdc_id;
'''
hql_allocationQttys = Template(hql_allocationQttys)


# 销量预测查询
hql_saleForecast = '''
drop table if exists dev.dev_tmp_lgb_saleForecast;
create table dev.dev_tmp_lgb_saleForecast as
select
    a.dt,
    a.fdcid,
    a.wid,
    b.forecast_daily_override_sales,
    c.total_sales,
    d.stock_qtty
from
    (
        select
            dt,
            wid,
            fdcid
        from
            dev.dev_inv_opt_fdc_sku_daily_summary_mid01
        where
            dt >= ${start_date} AND
            dt <=${end_date}  And
            fdcid=${dc_id}
    ) a
    left join
    (
        SELECT
            dt,
            dc_id,
            sku_id,
            forecast_begin_date,
            forecast_days,
            forecast_daily_override_sales  ---7天预测
        FROM
            app.app_pf_forecast_result_fdc_di  ---预测数据
        WHERE
            dt >= ${start_date} AND
            dt <=${end_date}  And
            dc_id=${dc_id}
     ) b
    on a.dt=b.dt
        and a.fdcid=b.dc_id
        and a.wid=b.sku_id
    left join
    (select
        dt,
        sku_id,
        dc_id,
        order_date,
        total_sales
    from
        app.app_sfs_sales_dc  ----FDC实际销量表
    where
        dt >= ${start_date} AND
        dt <=${end_date}  And
        dc_id=${dc_id}
    )  c
    on a.dt=c.dt
        and a.fdcid=c.dc_id
        and a.wid=c.sku_id
    left join
    (select
        dt,
        delv_center_num,
        sku_id,
        sum(stock_qtty+in_transit_qtty-sale_reserve_qtty)  as stock_qtty
    from
        gdm.gdm_m08_item_stock_day_sum   ---库存数据
    where
        dt between '$start_date' and '$end_date'
        and delv_center_num=${dc_id}
    group by
        dt,
        delv_center_num,
        sku_id) d
    on a.dt=d.dt
        and a.fdcid=d.delv_center_num
        and a.wid=d.sku_id;
'''
hql_saleForecast = Template(hql_saleForecast)


def hive_allocationQttys():
    '''
    查询调拨量
    '''
    os.system("hive -e '{0}'; ".format(hql_allocationQttys_create))
    date_range = getDate(start_date, end_date)
    for day in date_range:
        day = date_range[1]
        print day
        print "hive -e '{0}'  >>  allocationQttys.log; ".format(hql_allocationQttys.substitute(dt=day))
        os.system("hive -e '{0}'  >>  allocationQttys.log; ".format(hql_allocationQttys.substitute(dt=day)))
        exit()


def hive_salesForecast():
    '''
    查询预测销量
    '''
    os.system("hive -e '{0}'  >>  saleForecast.log; ".format(hql_saleForecast.substitute(start_date=start_date, end_date=end_date, dc_id=605)))


# 时间参数
start_date = '2016-12-01'
end_date = '2016-12-31'


if __name__ == '__main__':
    hive_allocationQttys()
    # hive_salesForecast()

