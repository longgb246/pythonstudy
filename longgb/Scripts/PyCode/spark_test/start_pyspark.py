#-*- coding:utf-8 -*-
# jdvl start -i bdp-docker.jd.com:5000/wise_mart_cmo_ipc -o='--net=host' -I bash
##### Use the Pyspark
# pyspark --master yarn  \
# --num-executors 15 \
# --executor-memory 15g \
# --executor-cores 4 \
# --driver-memory 20g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/jdk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/jdk1.8.0_121/jre/lib/amd64/server \
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

spark = SparkSession.builder.appName("test").getOrCreate()
# conf = SparkConf().setAppName('data hive2hdfs')
sc = SparkContext.getOrCreate()


mm = sc.parallelize([[23,4],[43,33]])
# sc.parallelize([[23,4],[43,33]]).toDF(['col1','col2']).show()
sp1 = spark.createDataFrame(mm,['col1','col2'])
mm = sc.parallelize([[23,4],[43,33]])
# sc.parallelize([[23,4],[43,33]]).toDF(['col1','col2']).show()
sp2 = spark.createDataFrame(mm,['col1','col3'])

sp1.show()
cc = sp1.collect()
print str(cc).replace('Row', '\nRow')

sp2.show()
sp3 = spark.createDataFrame(zip([23,23,4,4],['[43,33,332]','[43,33,332]','[43,33,332]','[43,33,332]']), ['col1','col3'])
# sp3.groupBy('col1').agg(F.udf(new_avg_udf)(F.collect_list('col3'), F.lit(3)).alias('tt')).show()
# sp3.select('col1', F.split('col3', ',')[1].alias('tmp')).show()
sp3.show()


sp1.unionAll(sp2).show()

type(mm)
type(sp)
isinstance(mm, RDD)
isinstance(sp, DataFrame)


def date_diff(x):
    if len(x) > 1:
        dt1 = datetime.datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S")
        delta_dt = (dt2 - dt1).__str__()
    else:
        delta_dt = '0:00:00'
    return delta_dt


mm2 = [['33', '2018-09-01 14:10:09'], ['32', '2018-09-01 14:10:19'], ['333', '2018-09-01 14:12:09'], ['34443', '2018-09-01 15:10:09']]
logger_sp = spark.createDataFrame(mm2, ['col1','dt'])
# logger_sp.show()
windowspec_r = Window.orderBy('dt').rowsBetween(Window.currentRow, 1)
logger_sp.select('col1', 'dt', F.udf(date_diff)(F.collect_list('dt').over(windowspec_r)).alias('date_diff')).orderBy(F.col('date_diff').desc()).show()
logger_sp.limit(2).rdd.collect()


print mm.keys().collect()
print mm.collect()
print 'finish!'
print sp.show()

