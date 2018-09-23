#-*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/2
  Usage   : 
"""

from pyspark.sql import SparkSession


spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sp1 = spark.table('app.dev_lgb_sku_tmp_band')

band_order = sp1.select('band').distinct().orderBy('band')
band_order.coalesce(1).write.csv('/user/mart_bca/longguangbin/sku_tmp_band_csv')

