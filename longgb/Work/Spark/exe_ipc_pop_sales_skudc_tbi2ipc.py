#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os



# hdfs_ipc_url = "hdfs://ns6"
# hdfs_tbi_url = "hdfs://172.22.100.100:8020"
# 
# table_name = "temp_pop_forecast_sales_skudc"
# 
# 
# ipc_table_path = "/user/mart_dm_tbi/forecast/result/pop/" + table_name
# tbi_table_path = "/user/mart_dm_tbi/forecast/result/pop/" + table_name
# 
# ipc_path = hdfs_ipc_url + ipc_table_path
# tbi_path = hdfs_tbi_url + tbi_table_path
# 
# 
# os.system('hadoop fs -rm -r -skipTrash'+ipc_path)
# os.system("hadoop fs -mkdir -p " + ipc_path)
# #os.system("hadoop fs -rm -r -skipTrash " + tbi_path)
# os.system("hadoop distcp " + tbi_path + " " + ipc_path)


## orders transform#################333


hdfs_ipc_url = "hdfs://ns6"
hdfs_tbi_url = "hdfs://172.22.100.100:8020"

table_name = "temp_pop_forecast_ords"


ipc_table_path = "/tmp/forecast/result/pop" 
tbi_table_path = "/tmp/forecast/result/pop/" + table_name

ipc_path = hdfs_ipc_url + ipc_table_path
tbi_path = hdfs_tbi_url + tbi_table_path


#os.system('hadoop fs -rm -r -skipTrash'+ipc_path)
#os.system("hadoop fs -mkdir -p " + ipc_path)
os.system("hadoop fs -rm -r -skipTrash " + tbi_path)
os.system("hadoop distcp " + tbi_path + " " + ipc_path)

###dc  ###################
hdfs_ipc_url = "hdfs://ns6"
hdfs_tbi_url = "hdfs://172.22.100.100:8020"

table_name = "temp_pop_forecast_ords_dc"


ipc_table_path = "/tmp/forecast/result/pop" 
tbi_table_path = "/tmp/forecast/result/pop/" + table_name

ipc_path = hdfs_ipc_url + ipc_table_path
tbi_path = hdfs_tbi_url + tbi_table_path


#os.system('hadoop fs -rm -r -skipTrash'+ipc_path)
#os.system("hadoop fs -mkdir -p " + ipc_path)
os.system("hadoop fs -rm -r -skipTrash " + tbi_path)
os.system("hadoop distcp " + tbi_path + " " + ipc_path)


