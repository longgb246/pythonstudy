#-*- coding:utf-8 -*-
import os
from string import Template

def pyhive(com_str):
    os.system('hive -e "{0}"  >>  test2.log  2>&1;'.format(com_str))


# 创建
hive_00 = '''set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
DROP TABLE IF EXISTS dev.dev_allocation_sku_data;
CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data
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

pyhive(hive_00)

