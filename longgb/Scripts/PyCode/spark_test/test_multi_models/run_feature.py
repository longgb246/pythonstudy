#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: "Zhangjianshen"
@contact: zhangjianshen@xxx.com
"""
from pyspark import SparkContext,SparkConf,SQLContext
from pyspark.sql import SparkSession
from src.pipeline.Pipeline import Pipeline
from src.config.param_config.param_config import param_dict
from src.utils.arg_parse import pipeline_arg_parse


##################################必须传入的参数###########################################
####读入常变参数
params = pipeline_arg_parse()
final_param_dict = param_dict
print(params)
print(param_dict)

###启动spark环境
try:
    sc.stop()
    conf = SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.memory.fraction",0.8)
    sc = SparkContext(conf).getOrCreate()               # 添加参数启动
except:
    conf = SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.memory.fraction",0.8)
    sc = SparkContext().getOrCreate()

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
Pp = Pipeline(sc, final_param_dict, spark=spark)
result = Pp.run_feature()
