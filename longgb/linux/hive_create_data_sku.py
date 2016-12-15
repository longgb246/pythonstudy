#-*- coding:utf-8 -*-
import os
from string import Template
import time

# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630/628/658 316

def printruntime(t1):
    d = time.time() - t1
    min_d = int(d / 60)
    sec_d = d % 60
    print 'Run Time is : {0}min {1:.4f}s'.format(min_d, sec_d)

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

# 创建
# hive_00 = '''set hive.exec.dynamic.partition=true;
# set hive.exec.dynamic.partition.mode=nonstrict;
# DROP TABLE IF EXISTS dev.dev_allocation_sku_data;
# CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data
# ( sku_id	 string,
#     forecast_begin_date	 string,
#     forecast_days	int,
#     forecast_daily_override_sales	array<double>,
#     forecast_weekly_override_sales	array<double>,
#     forecast_weekly_std	array<double>,
#     forecast_daily_std array<double>,
#     variance string,
#     ofdsales string,
#     inv	double,
#     arrive_quantity	int,
#     open_po	 int,
#     white_flag	 int,
#     white_flag_02  int)
#     PARTITIONED by (date_s  string,dc_id int);'''

hive_01 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid02_01_${dc_id}${test};
CREATE TABLE dev.tmp_allocation_order_pre_mid02_01_${dc_id}${test}
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
        delv_center_num='${org_dc_id}'
    group by
        delv_center_num,
        dt,
        sku_id;
'''
hive_01 = Template(hive_01)

hive_02 = '''DROP TABLE IF EXISTS dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id}${test};
CREATE TABLE dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id}${test}
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

hive_03 = '''DROP TABLE IF EXISTS dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id}${test};
CREATE TABLE dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id}${test}
AS
SELECT
        modify_time,
        create_time,
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

hive_04 = '''DROP TABLE IF EXISTS dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id}${test};
CREATE TABLE dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id}${test}
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

hive_05 = '''CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data${test}
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
    white_flag	 int,
    white_flag_02  int)
    PARTITIONED by (date_s  string,dc_id int);
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.max.dynamic.partitions=100000;
SET hive.exec.max.dynamic.partitions.pernode=100000;
INSERT OVERWRITE table dev.dev_allocation_sku_data${test}  partition(date_s,dc_id)
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
        case when c.modify_time is not null and c.modify_time<='2016-10-11' then 1
             when c.modify_time is not null and a.dt>c.modify_time then 1
             else 0 end as white_flag,
        case when c.create_time is not null and a.dt>c.create_time then 1 else 0 end as white_flag_02,
        a.dt as date_s,
        b.dc_id
    FROM
        dev.tmp_allocation_order_pre_mid02_01_${dc_id}${test} a
    LEFT JOIN
        dev.tmp_app_pf_forecast_result_fdc_di_01_${dc_id}${test} b
    ON
        a.sku_id=b.sku_id AND
        a.dt=b.dt
    LEFT JOIN
        dev.tmp_fdm_fdc_whitelist_chain_01_${dc_id}${test} c
    ON
        a.sku_id=c.wid
    LEFT JOIN
        dev.tmp_app_sfs_rdc_forecast_result_01_${dc_id}${test} d
    ON
        a.sku_id=d.sku_id AND
        a.dt=d.dt;
'''
hive_05 = Template(hive_05)


hive_select = '''select * from dev.${table}${dc_id}${test} limit 10;
'''
hive_select = Template(hive_select)

hive_drop = '''DROP TABLE IF EXISTS dev.dev_allocation_sku_data${test};
'''
hive_drop = Template(hive_drop)

# 2016-07-01 2016-11-01 630 316
start_date = '2016-07-01'
end_date = '2016-11-01'
dc_id_list = ['630','628','658']
drop_table = 0      # 为 0 的时候，表示不删除表， 为 1 ，删除表
istest = '_test'    # 为 ' ' 的时候，表示正式插入表，有任何字符表示创建 test 的表


if __name__ == '__main__':
    if drop_table == 1:
        os.system('hive -e "{0}";'.format(hive_drop.substitute(test=istest)))
    for each in dc_id_list:
        print "{0} ...".format(each)
        t1 = time.time()
        print 'Table:tmp_allocation_order_pre_mid02_01_{0}{1}   '.format(each, istest),
        pyhive(hive_01.substitute(start_date=start_date,end_date=end_date,dc_id=each,org_dc_id=316,test=istest), 'sku_data_{0}.log'.format(each))
        pyhive(hive_select.substitute(table='tmp_allocation_order_pre_mid02_01_', dc_id=each,test=istest), 'sku_data_{0}_select.log'.format(each))
        printruntime(t1)
        t1 = time.time()
        print 'Table:tmp_app_pf_forecast_result_fdc_di_01_{0}{1}   '.format(each, istest),
        pyhive(hive_02.substitute(start_date=start_date,end_date=end_date,dc_id=each,test=istest), 'sku_data_{0}.log'.format(each))
        pyhive(hive_select.substitute(table='tmp_app_pf_forecast_result_fdc_di_01_', dc_id=each,test=istest), 'sku_data_{0}_select.log'.format(each))
        printruntime(t1)
        t1 = time.time()
        print 'Table:tmp_fdm_fdc_whitelist_chain_01_{0}{1}   '.format(each, istest),
        pyhive(hive_03.substitute(start_date=start_date,end_date=end_date,dc_id=each,test=istest), 'sku_data_{0}.log'.format(each))
        pyhive(hive_select.substitute(table='tmp_fdm_fdc_whitelist_chain_01_', dc_id=each,test=istest), 'sku_data_{0}_select.log'.format(each))
        printruntime(t1)
        t1 = time.time()
        print 'Table:tmp_app_sfs_rdc_forecast_result_01_{0}{1}   '.format(each, istest),
        pyhive(hive_04.substitute(start_date=start_date,end_date=end_date, dc_id=each, org_dc_id=316,test=istest), 'sku_data_{0}.log'.format(each))
        pyhive(hive_select.substitute(table='tmp_app_sfs_rdc_forecast_result_01_', dc_id=each,test=istest), 'sku_data_{0}_select.log'.format(each))
        printruntime(t1)
        t1 = time.time()
        print 'Create Table : dev.dev_allocation_sku_data{0};  '.format(istest),
        pyhive(hive_05.substitute(dc_id=each, test=istest), 'sku_data_{0}.log'.format(each))
        printruntime(t1)



# # test
# path = r'D:\Lgb\rz_sz\000000_0'
# import pandas as pd
# data = pd.read_table(path, sep='\001', header=None)
# def getvalue_map(x):
#     data = x.split('\002')
#     data2 = np.array(map(lambda x: float(x),data))
#     return data2
# data3 = map(getvalue_map,data[3].values)
# mm = data[3][0].split('\002')
# mm2 = data[3][0].split('\x02')
