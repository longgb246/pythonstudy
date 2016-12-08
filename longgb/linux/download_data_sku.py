#-*- coding:utf-8 -*-
import os
from string import Template
import time

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*30, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*30, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

# 创建
hive_00 = '''set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
DROP TABLE IF EXISTS dev.dev_allocation_sku_data_${dc_id};
CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data_${dc_id}
( sku_id	 string,
    forecast_begin_date	 string,
    forecast_days	int,
    forecast_daily_override_sales	array<double>,
    forecast_weekly_override_sales	array<double>,
    forecast_weekly_std	array<double>,
    forecast_daily_std array<double>,
    variance string,
    ofdsales string,
    inv	double,
    arrive_quantity	int,
    open_po	 int,
    white_flag	 int)
    PARTITIONED by (date_s  string,dc_id int);'''
hive_00 = Template(hive_00)

hive_01 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid02_01_${dc_id};
CREATE TABLE dev.tmp_allocation_order_pre_mid02_01_${dc_id}
AS
SELECT
        delv_center_num,
        dt,
        sku_id,
        sum(in_transit_qtty) AS open_po,
        sum(stock_qtty) AS inv --库存数量
    FROM
        gdm.gdm_m08_item_stock_day_sum
    WHERE
        dt>='${start_date}' AND
        dt<='${end_date}' AND
        delv_center_num='${dc_id}'
    group by
        delv_center_num,
        dt,
        sku_id;
'''
hive_01 = Template(hive_01)

hive_02 = '''DROP TABLE IF EXISTS dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id};
CREATE TABLE dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id}
AS
SELECT
        dt,
        dc_id,
        sku_id,
        forecast_begin_date,
        forecast_days,
        forecast_daily_override_sales ,
        forecast_weekly_override_sales,
        forecast_weekly_std,
        forecast_daily_std--新增每日
    FROM
        app.app_pf_forecast_result_fdc_di
    WHERE
        dt >= '${start_date}' AND
        dt <= '${end_date}'  And
        dc_id= '${dc_id}';
'''
hive_02 = Template(hive_02)

hive_03 = '''DROP TABLE IF EXISTS dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id};
CREATE TABLE dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id}
AS
SELECT
        modify_time,
        wid,
        fdcid
    FROM
        fdm.fdm_fdc_whitelist_chain
    WHERE
        start_date<= '${end_date}' AND
        end_date>= '${end_date}' AND
        yn= 1 AND
        fdcid= '${dc_id}';
'''
hive_03 = Template(hive_03)

hive_04 = '''DROP TABLE IF EXISTS dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id};
CREATE TABLE dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id}
AS
SELECT
      dt,
        wid as sku_id,
        dcid,
        variance,
        ofdsales
        from app.app_sfs_rdc_forecast_result
    WHERE
        dt >= '${start_date}' AND
        dt <= '${end_date}'  And
        dcid= '${org_dc_id}';
'''
hive_04 = Template(hive_04)

hive_05 = '''INSERT OVERWRITE table dev.dev_allocation_sku_data_${dc_id}  partition(date_s,dc_id)
SELECT
    a.sku_id,
    b.forecast_begin_date,
    b.forecast_days,
    b.forecast_daily_override_sales,
    b.forecast_weekly_override_sales,
    b.forecast_weekly_std,
    b.forecast_daily_std,
    d.variance,
    d.ofdsales,
    a.inv,
    0 as arrive_quantity, --这个会在程序中更新
    a.open_po,
    case when c.modify_time<a.dt then 1 else 0 end as white_flag,
    a.dt as date_s,
    b.dc_id
FROM
    dev.tmp_allocation_order_pre_mid02_01_${dc_id} a
LEFT JOIN
    dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id} b
ON
    a.sku_id=b.sku_id AND
    a.dt=b.dt
LEFT JOIN
    dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id} c
ON
    a.sku_id=c.wid
LEFT JOIN
    dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id} d
ON
    a.sku_id=d.sku_id AND
    a.dt=d.dt;
'''
hive_05 = Template(hive_05)

hive_select = '''select * from dev.${table}${dc_id} limit 10;
'''
hive_select = Template(hive_select)

# 2016-07-01 2016-11-01 630 316
dc_id_list = ['630','628','658']
for each in dc_id_list:
    print "{0} ...".format(each)
    t1 = time.time()
    pyhive(hive_00.substitute(dc_id=each), 'sku_data.log')
    pyhive(hive_01.substitute(start_date='2016-07-01',end_date='2016-11-01',dc_id=each), 'sku_data_{0}.log'.format(each))
    pyhive(hive_select.substitute(table='tmp_allocation_order_pre_mid02_01_', dc_id=each), 'sku_data_{0}_select.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)

    t1 = time.time()
    pyhive(hive_02.substitute(start_date='2016-07-01',end_date='2016-11-01',dc_id=each), 'sku_data_{0}.log'.format(each))
    pyhive(hive_select.substitute(table='tmp_app_pf_forecast_result_fdc_di_01_', dc_id=each), 'sku_data_{0}_select.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)

    t1 = time.time()
    pyhive(hive_03.substitute(start_date='2016-07-01',end_date='2016-11-01',dc_id=each), 'sku_data_{0}.log'.format(each))
    pyhive(hive_select.substitute(table='tmp_fdm_fdc_whitelist_chain_01_', dc_id=each), 'sku_data_{0}_select.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)

    t1 = time.time()
    pyhive(hive_04.substitute(start_date='2016-07-01',end_date='2016-11-01', dc_id=each, org_dc_id=316), 'sku_data_{0}.log'.format(each))
    pyhive(hive_select.substitute(table='tmp_app_sfs_rdc_forecast_result_01_', dc_id=each), 'sku_data_{0}_select.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)

    pyhive(hive_05.substitute(dc_id=each), 'sku_data_{0}.log'.format(each))



