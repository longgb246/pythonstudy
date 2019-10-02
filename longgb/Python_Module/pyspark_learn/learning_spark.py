# -*- coding:utf-8 -*-
# ===============================================================================
#
#          FILE: LearningSpark.py
#
#   DESCRIPTION: Spark Learning
#       OPTIONS: ---
#  REQUIREMENTS: ---
#         NOTES: ---
#        AUTHOR: longguangbin@163.com
#       VERSION: 1.0
#       CREATED: 2017-11-06
#        MODIFY: ---
# ===============================================================================
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
import os

# 一、SparkSession 部分  SQL、DataFrame 部分
# spark = SparkSession.builder \            # SparkSession 类中的 Builder 类
#     .master("local") \                    # 运行模式
#     .appName("TimingLimit") \
#     .config("spark.some.config.option", "some-value") \
#     .getOrCreate()                        # 获得或者创建一个 SparkSession 类
spark = SparkSession.builder \
    .master("yarn") \
    .appName("TimingLimit") \
    .getOrCreate()

order_info = spark.table('dev.dev_lgb_fullStock_TimingLimit_order_info')
waybill_info = spark.table('dev.dev_lgb_fullStock_TimingLimit_waybill_info')


# dev_lgb_fullStock_TimingLimit_order_info
# dev_lgb_fullStock_TimingLimit_waybill_info


# 二、SparkSession 部分     rdd 部分
def rm_dir_files(rm_path):
    """
    saveAsTextFile 方法不能覆盖文件夹，需要先删除。
    """
    rm_cmd = 'hadoop fs -rm -r {0}'.format(rm_path)
    try:
        os.system(rm_cmd)
    except:
        print('[ {0} ] the path has already been removed !'.format(rm_cmd))


save_path = r'hdfs://ns15/user/cmo_ipc/longguangbin/work/pytest'

sc = SparkContext(master="yarn", appName="My App")
sc_conf = map(lambda x: x[0] + ':' + x[1], sc.getConf().getAll())
rm_dir_files(save_path + os.sep + 'sc_conf')
sc.parallelize(sc_conf).repartition(1).saveAsTextFile(save_path + os.sep + 'sc_conf')
