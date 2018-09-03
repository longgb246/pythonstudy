#-*- coding:utf-8 -*-
import findspark
# findspark.init(r'D:\Softwares\pyspark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\bin')
findspark.init("/Users/longguangbin/SoftWare/spark-2.1.0-bin-hadoop2.7")

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
# conf = SparkConf().setAppName('data hive2hdfs')
sc = SparkContext.getOrCreate()


mm = sc.parallelize([[23,4],[43,33]])
sp = spark.createDataFrame(mm,['col1','col2'])


def transpose(matrix):
    return map(list, zip(*matrix))


from string import Template
len_tem.substitute(len=len_col[0])


def printContent(sp):
    len_tem = Template('{:${len}}')
    sp_columns = sp.columns
    sp_list = map(lambda x: map(str, list(x)), sp.collect())
    sp_all_list = [sp_columns]+sp_list
    len_col = map(lambda x: max(map(lambda y: len(y),x))+1, transpose(sp_all_list))
    map(lambda x: map(lambda i: len_tem.substitute(x), range(len(len_col))), sp_all_list)
    # print ', '.join(len_col)
    for each in sp_list:
        print(each)


print mm.keys().collect()
print mm.collect()
print 'finish!'
print sp.show()


sp1 = spark.createDataFrame([["a", 1],["b",1],["c",1]], ["sku", "value"])
sp2 = spark.createDataFrame([["a", 1],["b",2],], ["sku", "value2"])
sp1.join(sp2, on=["sku"], how="left").na.fill(0).show()


sp.printSchema()
# sp.schema
sp.count()
sp.rdd.getNumPartitions()

