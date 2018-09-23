# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/13
  Usage   : 
"""

# 目前 dev 的日期是 30-31-1 (2-3-4) 5-6-7-8-10-11

# 07-30 - dev2   - all_gap : 87135.89179999994,  all_real : 78248.0, mape : 1.113586184950413, kpi: 0.998347796795

# 07-30 - online - all_gap : 136006.3671785484,  all_real : 78248.0, mape : 1.738144964453384, kpi: 0.999320397987
# 07-30 - dev    - all_gap : 115193.2301000001,  all_real : 78248.0, mape : 1.472155583529293, kpi: 0.999320397987

# ***
# 07-31 - online - all_gap : 144382.381021158,   all_real : 74409.0, mape : 1.940388676385368, kpi: 0.99954951734
# 07-31 - dev    - all_gap : 115896.384500000,   all_real : 74409.0, mape : 1.557558689137067, kpi: 0.999563818377
# 07-31 - dev2   - all_gap : 113220.840200000,   all_real : 74407.0, mape : 1.521642321286978, kpi: 0.999510841734

# 08-01 - online - all_gap : 144357.7450857223,  all_real : 74143.0, mape : 1.947017858539880, kpi: 0.999557619996
# 08-01 - dev    - all_gap : 125477.7349999998,  all_real : 74143.0, mape : 1.692374667871544, kpi: 0.999557619996

# 08-02 - online - all_gap : 145231.8739679934,  all_real : 74635.0, mape : 1.945895008615172, kpi: 0.999616876264
# 08-02 - dev    - all_gap : 119832.4678,        all_real : 74635.0, mape : 1.605580060293428, kpi: 0.999616876264


from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# 先拿 31 号来分析。
gap_sp, all_mape_sp, kpi = spark.createDataFrame(), spark.createDataFrame(), spark.createDataFrame()

gap_sp.show()
gap_sp.orderBy('gap', ascending=False).show()
mm = gap_sp.orderBy('gap', ascending=False).take(20)

import pandas as pd

gap_sp.where(''' gap < 30 ''').agg(F.sum('real_sale').alias('real'), F.sum('gap').alias('gap')).withColumn('mape', F.col('gap')/F.col('real')).show()
gap_sp.agg(F.sum('real_sale').alias('real'), F.sum('gap').alias('gap')).withColumn('mape', F.col('gap')/F.col('real')).show()


gap_sp.groupBy('sku_id').agg(F.sum(F.col('real_sale')).alias('real_sale'), F.sum(F.col('pre_sale')).alias('pre_sale')).\
    withColumn('gap', F.abs(F.col('real_sale') - F.col('pre_sale'))).\
    agg(F.sum(F.col('real_sale')).alias('real'), F.sum(F.col('gap')).alias('gap')).\
    withColumn('mape', F.col('gap')/F.col('real')).show()



gap_sp.where(''' mape != 1 and mape != 0 ''').show()


sample_df = pd.DataFrame.from_records(map(lambda x: x.asDict(), mm))

gap_sp.count()
# 832443

gap_sp.where(''' gap > 20 or mape > 0.7 ''').count()
# 52105

# 52105*20

# XSTD-1：(塑料袋)
# select DISTINCT sku_code, store_id, channel_id from app.app_saas_sfs_model_input where sku_code = 'XSTD-1'
# 1、XSTD-1 - K511：存在 sku 突增 + 突降 情况，这里需要把 突增 + 突降 情况考虑进去
# 2、

# L64T



#         dt         gap        mape     pre_sale    real_sale    sku_id
# 0   2018-08-01  396.7935    0.977324    9.2065      406.0     XSTD-1_K511_90_ZP008#XSTD-1#K511#90
# 2   2018-08-05  269.1447    0.941065   16.8553      286.0     XSTD-1_K511_90_ZP008#XSTD-1#K511#90
# 3   2018-08-04  242.9568    0.964114    9.0432      252.0     XSTD-1_K511_90_ZP008#XSTD-1#K511#90

# 1   2018-08-04  300.0000    1.000000    0.0000      300.0     XSTD-1_K50W_90_ZP008#XSTD-1#K50W#90

# 7   2018-08-05  171.3272    0.744901   58.6728      230.0     XSTD-1_K50M_90_ZP008#XSTD-1#K50M#90

# 4   2018-08-04  200.0000    1.000000    0.0000      200.0     XSTD-1_L606_90_ZP008#XSTD-1#L606#90

# 9   2018-08-05  159.6225    0.840118   30.3775      190.0     XSTD-1_K55G_90_ZP008#XSTD-1#K55G#90

# 10  2018-08-07  128.3333    0.000000  128.3333        0.0     XSTD-1_L64T_90_ZP008#XSTD-1#L64T#90
# 11  2018-08-06  115.1102    0.000000  115.1102        0.0     XSTD-1_L64T_90_ZP008#XSTD-1#L64T#90
# 12  2018-08-03  112.1769    0.000000  112.1769        0.0     XSTD-1_L64T_90_ZP008#XSTD-1#L64T#90
# 17  2018-08-04   98.6842    0.000000   98.6842        0.0     XSTD-1_L64T_90_ZP008#XSTD-1#L64T#90
# 14  2018-08-05  108.4232    0.000000  108.4232        0.0     XSTD-1_L64T_90_ZP008#XSTD-1#L64T#90

# 15  2018-08-07  107.3236    0.761160   33.6764      141.0     XSTD-1_L64C_90_ZP008#XSTD-1#L64C#90

# 16  2018-08-06  100.0000    1.000000    0.0000      100.0     XSTD-1_K53H_90_ZP008#XSTD-1#K53H#90

# 5   2018-08-07  193.4234    0.000000  193.4234        0.0     FSTD-3_L64T_90_ZP008#FSTD-3#L64T#90
# 6   2018-08-04  191.7751    0.000000  191.7751        0.0     FSTD-2_K50W_90_ZP008#FSTD-2#K50W#90
# 8   2018-08-04  162.7142  162.714200  163.7142        1.0     FSTD-3_L64T_90_ZP008#FSTD-3#L64T#90
# 13  2018-08-05  110.9394    0.000000  110.9394        0.0     FSTD-2_K50W_90_ZP008#FSTD-2#K50W#90
# 18  2018-08-03   98.5540   49.277000  100.5540        2.0     FSTD-3_L64T_90_ZP008#FSTD-3#L64T#90
# 19  2018-08-03   97.7657    0.763795   30.2343      128.0     FSTD-2_K508_90_ZP008#FSTD-2#K508#90

gap_sp.where(''' gap > 20 or mape > 0.7 ''').groupBy('sku_id')

gap_sku_sp = gap_sp.groupBy('sku_id').agg(F.sum('gap').alias('gap'), F.sum('real_sale').alias('real_sale'), F.sum('pre_sale').alias('pre_sale'))

mm = gap_sku_sp.orderBy('gap', ascending=False)
focus_sp = mm.where('''  ''')
# 139850 为 sku 数目


