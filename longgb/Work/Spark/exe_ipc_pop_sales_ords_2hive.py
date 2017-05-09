#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import datetime
sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask

ht = HiveTask()

_today = str(datetime.date.today())
# sql = """
#     use app;
#     set hive.exec.max.created.files    = 655350;
#     set mapred.max.split.size          = 100000000;
#     set mapred.min.split.size.per.node = 100000000;
#     set mapred.min.split.size.per.rack = 100000000;
#     set hive.input.format              =
#     org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
#     set mapred.output.compress                   = true;
#     set hive.auto.convert.join                   = true;
#     set hive.exec.dynamic.partition              = true;
#     set hive.exec.dynamic.partition.mode         = nonstrict;
#     set hive.exec.max.dynamic.partitions         = 100000;
#     set hive.exec.max.dynamic.partitions.pernode = 100000;
#     set hive.merge.mapfiles                      = true;
#     set hive.merge.mapredfiles                   = true;
#     set hive.merge.size.per.task                 = 256000000;
#     set hive.merge.smallfiles.avgsize            = 64000000;
#     set mapred.reduce.tasks                      = 200;
#     LOAD DATA INPATH '/tmp/forecast/result/pop/temp_pop_forecast_sales_skudc' OVERWRITE INTO TABLE app.app_sfs_pop_sales_ords_forecast PARTITION( dt  = '"""+ _today +"""',key = 'sales_sku_dc');
# 	"""
# print(sql)
# ht.exec_sql(schema_name = 'app', table_name = 'app_sfs_pop_sales_ords_forecast', sql = sql )



sql_01 = """
    use app;
    set hive.exec.max.created.files    = 655350;
    set mapred.max.split.size          = 100000000;
    set mapred.min.split.size.per.node = 100000000;
    set mapred.min.split.size.per.rack = 100000000;
    set hive.input.format              =
    org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
    set mapred.output.compress                   = true;
    set hive.auto.convert.join                   = true;
    set hive.exec.dynamic.partition              = true;
    set hive.exec.dynamic.partition.mode         = nonstrict;
    set hive.exec.max.dynamic.partitions         = 100000;
    set hive.exec.max.dynamic.partitions.pernode = 100000;
    set hive.merge.mapfiles                      = true;
    set hive.merge.mapredfiles                   = true;
    set hive.merge.size.per.task                 = 256000000;
    set hive.merge.smallfiles.avgsize            = 64000000;
    set mapred.reduce.tasks                      = 200;
    LOAD DATA INPATH '/tmp/forecast/result/pop/temp_pop_forecast_ords' OVERWRITE INTO TABLE app.app_sfs_pop_sales_ords_forecast PARTITION( dt  = '"""+ _today +"""',key = 'orders_vender');
	"""   
print(sql_01)
ht.exec_sql(schema_name = 'app', table_name = 'app_sfs_pop_sales_ords_forecast', sql = sql_01 )


sql_02 = """
    use app;
    set hive.exec.max.created.files    = 655350;
    set mapred.max.split.size          = 100000000;
    set mapred.min.split.size.per.node = 100000000;
    set mapred.min.split.size.per.rack = 100000000;
    set hive.input.format              =
    org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
    set mapred.output.compress                   = true;
    set hive.auto.convert.join                   = true;
    set hive.exec.dynamic.partition              = true;
    set hive.exec.dynamic.partition.mode         = nonstrict;
    set hive.exec.max.dynamic.partitions         = 100000;
    set hive.exec.max.dynamic.partitions.pernode = 100000;
    set hive.merge.mapfiles                      = true;
    set hive.merge.mapredfiles                   = true;
    set hive.merge.size.per.task                 = 256000000;
    set hive.merge.smallfiles.avgsize            = 64000000;
    set mapred.reduce.tasks                      = 200;
    LOAD DATA INPATH '/tmp/forecast/result/pop/temp_pop_forecast_ords_dc' OVERWRITE INTO TABLE app.app_sfs_pop_sales_ords_forecast PARTITION( dt  = '"""+ _today +"""',key = 'orders_vender_dc');
	"""   
print(sql_02)
ht.exec_sql(schema_name = 'app', table_name = 'app_sfs_pop_sales_ords_forecast', sql = sql_02 )