# -*- coding:utf-8 -*-
# xxxvl start -i bdp-docker.xxx.com:5000/wise_mart_cmo_ipc -o='--net=host' -I bash
##### Use the Pyspark
# pyspark --master yarn  \
# --num-executors 15 \
# --executor-memory 15g \
# --executor-cores 4 \
# --driver-memory 20g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --files $HIVE_CONF_DIR/hive-site.xml


import findspark

# findspark.init(r'D:\Softwares\pyspark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\bin')
# findspark.init(r'D:\Softwares\pyspark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7')
findspark.init("/Users/longguangbin/SoftWare/spark-2.1.0-bin-hadoop2.7")

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import Window
import pyspark.sql.functions as F

import datetime

# mlp
# spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
# local
spark = SparkSession.builder.appName("test").getOrCreate()
# conf = SparkConf().setAppName('data hive2hdfs')
sc = SparkContext.getOrCreate()

mm = sc.parallelize([[23, 4], [43, 33]])
# sc.parallelize([[23,4],[43,33]]).toDF(['col1','col2']).show()
sp1 = spark.createDataFrame(mm, ['col1', 'col2'])
mm = sc.parallelize([[23, 4], [43, 33]])
# sc.parallelize([[23,4],[43,33]]).toDF(['col1','col2']).show()
sp2 = spark.createDataFrame(mm, ['col1', 'col3'])

sp1.show()
sp1_filter = sp1.where(F.col('col1') > 100)
len(sp1_filter.head(1))

sp4 = spark.createDataFrame([['aa|11', 'bb'], ['cc|dd', 'ee']], ['a', 'b'])
sp4.show()
sp4.withColumn('d', F.split(F.col('a'), '\|')[1]).show()

sp5 = spark.createDataFrame([['aa', 'bb', '2018-01-01'],
                             ['cc', 'ee', '2018-01-02'],
                             ['aa', 'b', '2018-01-03'],
                             ['dd', 'b', '2018-01-03']],
                            ['a', 'b', 'dt'])

sp5.show()
sp5.select(F.max('dt').alias('end_date')).collect()[0]['end_date']


def spark_get_first(sp, partition_by, order_y):
    order_y = [F.col(x) for i, x in enumerate(order_y)]
    # order_y = [F.col(x).desc() for i, x in enumerate(order_y)]
    window = Window.partitionBy(partition_by).orderBy(*order_y)

    first_sp = sp.select('*', F.row_number().over(window).alias('row_number')). \
        where(F.col('row_number') <= 1). \
        drop('row_number')
    # df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 2)
    return first_sp


sp5.show()

sp4 = spark.createDataFrame([[1.0, 2.0, 9.0], [2.0, 4.0, 10.0], [1.0, 2.0, 9.0]],
                            ['_granu_', '_order_', '_valid_cols_'])
sp4.show()
sp4.distinct().show()

dt1 = '2018-01-03'
dt2 = '2018-01-01'

# dt2 = (datetime.datetime.strptime(dt1, '%Y-%m-%d') - datetime.timedelta(7)).strftime('%Y-%m-%d')
filter_sp = sp5.where(''' dt != '{dt1}' '''.format(dt1=dt1))
# replace_sp = sp.where(''' dt = '{dt2}' '''.format(dt2=dt2)).withColumn('dt', F.lit('{dt1}'.format(dt1=dt1)))
fill_sp_1 = sp5.where(''' dt = '{dt1}' '''.format(dt1=dt1)).withColumn('b', F.lit(0.0))
fill_sp_2 = sp5.where(''' dt = '{dt2}' '''.format(dt2=dt2))
sp7_1 = fill_sp_1.union(fill_sp_2)
sp7_1.show()
sp7_2 = spark_get_first(sp7_1, ['a'], ['dt'])
sp7_2.show()
replace_sp = sp7_2.withColumn('dt', F.lit(dt1))
sp7 = filter_sp.union(replace_sp)
sp7.show()

replace_sp = sp5.where(''' dt = '{dt1}' or dt = '{dt2}' '''.format(dt1=dt1, dt2=dt2))

sp5_1 = sp5.where(''' dt = '{dt1}' or dt = '{dt2}' '''.format(dt1=dt1, dt2=dt2))

sp5_1.show()

# stop


replace_sp = sp5.where(''' dt = '{dt2}' '''.format(dt2=dt2)).withColumn('dt', F.lit('{dt1}'.format(dt1=dt1)))

sku_sp = sp5.where(''' dt = '{dt1}' or dt = '{dt2}' '''.format(dt1=dt1, dt2=dt2)).select('a').distinct()
sku_sp.show()

sku_sp.join(replace_sp, on=['a'], how='left').show()

filter_sp.show()
replace_sp.show()

sp5.filter(F.col('mm') == 1).show()
sp5.filter('mm').show()

sp5.show()

dt1 = '2018-01-02'
dt2 = '2018-01-03'
filter_sp = sp5.where(''' dt != '{dt1}' '''.format(dt1=dt1))
replace_sp = sp5.where(''' dt = '{dt1}' '''.format(dt1=dt1)).withColumn('dt', F.lit('{dt2}'.format(dt2=dt2)))
sp6 = filter_sp.union(replace_sp)
sp6.show()

dt1 = '2018-12-26'
(datetime.datetime.strptime(dt1, '%Y-%m-%d') - datetime.timedelta(7)).strftime('%Y-%m-%d')

sp5.show()
sp6 = spark.createDataFrame([['aa', 'b1', '2018-01-01'],
                             ['cc', 'ee2', '2018-01-02'],
                             ['cc', 'ee3', '2018-01-02'],
                             ['cd', 'e2e', '2018-01-04'],
                             ['cd', 'b3', '2018-01-03']], ['a', 'b', 'dt'])
sp6.show()
# mm = sp5.join(sp6, (sp5['a'] == sp6['a']) & (sp5['dt'] == sp6['dt']), 'left')
mm = sp5.join(sp6, ['a', 'dt'], 'left')
mm.drop(sp6['a']).show()

mm.withColumnRenamed('a', 'as').show()

df_as1 = sp5.alias("df_as1")
df_as2 = sp6.alias("df_as2")
mm = df_as1.join(df_as2, (df_as1['a'] == df_as2['a']) & (df_as1['dt'] == df_as2['dt']), 'left')
mm.drop(df_as2['a']).show()
mm.drop('b').show()


def spark_get_first(sp, partition_by, order_y):
    order_y = [F.col(x).desc() for x in order_y]
    window = Window.partitionBy(partition_by).orderBy(*order_y)

    first_sp = sp.select('*', F.row_number().over(window).alias('row_number')). \
        where(F.col('row_number') <= 1). \
        drop('row_number')
    # df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 2)
    return first_sp


sp5.show()
spark_get_first(sp5, ['a'], ['dt']).show()

cc = sp1.collect()
print(str(cc).replace('Row', '\nRow'))

from pyspark.sql.types import *

schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", StringType(), True)])

sp4.rdd.toDF(schema).show()

sp4.show()
sp4.rdd.getNumPartitions()


def pp(x):
    print(x)


sp4.foreachPartition(lambda x: pp(x))
sp4.rdd.mapPartitions(lambda x: pp(x))

sp2.show()
sp3 = spark.createDataFrame(zip([23, 23, 4, 4], ['[43,33,332]', '[43,33,332]', '[43,33,332]', '[43,33,332]']),
                            ['col1', 'col3'])
# sp3.groupBy('col1').agg(F.udf(new_avg_udf)(F.collect_list('col3'), F.lit(3)).alias('tt')).show()
# sp3.select('col1', F.split('col3', ',')[1].alias('tmp')).show()
sp3.show()

sp4 = spark.createDataFrame([[1, 2, 3], [3, 4, 5], [6, 7, 8]], ['a', 'b', 'c'])
sp4.show()
sp4.drop(*['a', 'b']).show()

sp1.unionAll(sp2).show()

type(mm)
type(sp1)
isinstance(mm, RDD)
isinstance(sp1, DataFrame)


def date_diff(x):
    if len(x) > 1:
        dt1 = datetime.datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S")
        delta_dt = (dt2 - dt1).__str__()
    else:
        delta_dt = '0:00:00'
    return delta_dt


mm2 = [['33', '2018-09-01 14:10:09'], ['32', '2018-09-01 14:10:19'], ['333', '2018-09-01 14:12:09'],
       ['34443', '2018-09-01 15:10:09']]
logger_sp = spark.createDataFrame(mm2, ['col1', 'dt'])
# logger_sp.show()
windowspec_r = Window.orderBy('dt').rowsBetween(Window.currentRow, 1)
logger_sp.select('col1', 'dt', F.udf(date_diff)(F.collect_list('dt').over(windowspec_r)).alias('date_diff')).orderBy(
    F.col('date_diff').desc()).show()
logger_sp.limit(2).rdd.collect()

print mm.keys().collect()
print mm.collect()
print 'finish!'
print sp1.show()

sp1.show()
sp1.where(F.col('col1').isin([23])).show()
