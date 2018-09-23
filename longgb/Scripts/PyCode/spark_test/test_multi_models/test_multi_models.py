#-*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/2
  Usage   : 
"""    

from dateutil.parser import parse
import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window


spark = SparkSession.builder.enableHiveSupport().getOrCreate()

df1 = spark.sql(''' select * from app.app_saas_sfs_model_input where dt='2018-07-31' ''').select(['sku_code', 'sale_date', 'sale'])
df1.show()

day_len = 90
day_end = '2018-07-31'
day_start = (parse(day_end) - datetime.timedelta(day_len)).strftime('%Y-%m-%d')

df1_sum = df1.where(''' sale_date >= '{day_start}' and sale_date <= '{day_end}' '''.format(day_start=day_start, day_end=day_end)).groupBy('sku_code').agg(F.sum('sale').alias('sale_sum'))
# Temp rank of the sale, just to split into 5
windowspec_r = Window.orderBy(F.col('sale_sum').desc())
df1_rank = df1_sum.withColumn('rank', F.rank().over(windowspec_r))
# 16483
df1_cnt = df1_sum.select(F.countDistinct('sku_code').alias('sku_count'))

df1_rcnt = df1_rank.crossJoin(F.broadcast(df1_cnt))
df1_rcnt = df1_rcnt.withColumn('rank_rate', F.col('rank')/ F.col('sku_count'))

band_sql = '''
Case
    When rank_rate < 0.2 Then 1
    When rank_rate < 0.4 Then 2
    When rank_rate < 0.6 Then 3
    When rank_rate < 0.8 Then 4
    else 5
end
as
    band
'''

df1_band = df1_rcnt.selectExpr('sku_code', 'rank', band_sql)

df1_band.write.saveAsTable('app.dev_lgb_sku_tmp_band')
# df1_band.write.csv('/user/mart_bca/longguangbin/sku_tmp_band_csv')

