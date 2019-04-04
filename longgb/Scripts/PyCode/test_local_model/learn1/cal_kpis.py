# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/8
  Usage   : 
"""

#### xxxvl start -m /data0/mart_bca:/data0/mart_bca:rw -i bdp-docker.xxx.com:5000/wise_mart_bca:latest -o='--net=host' -I bash
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server

# spark-submit --master yarn --deploy-mode cluster \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/yarn-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/yarn-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# test_little_files.py

from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY
import datetime
from string import Template
import pandas as pd
import pprint

from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkConf


# sp_conf = SparkConf()
# sp_conf.set("spark.sql.hive.mergeFiles", "true")
# conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
# conf.set("spark.memory.fraction",0.8)
# sc = SparkContext().getOrCreate()               # 添加参数启动

def createSparkSession(appName):
    conf = SparkConf().setAppName(appName)
    # conf.set("spark.dynamicAllocation.enabled","true")
    conf.set("spark.rdd.compress", "true")
    conf.set("spark.broadcast.compress", "true")
    conf.set("hive.exec.dynamic.partition", "true")
    conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
    conf.set("hive.exec.max.dynamic.partitions", "100000")
    conf.set("hive.exec.max.dynamic.partitions.pernode", "100000")
    conf.set("hive.auto.convert.join", "true")
    #     1. Map输入合并小文件--没起作用
    conf.set("mapred.max.split.size", "256000000")  # 每个Map最大输入大小
    conf.set("mapred.min.split.size.per.node", "100000000")  # 一个节点上split的至少的大小
    conf.set("mapred.min.split.size.per.rack", "100000000")  # 一个交换机下split的至少的大小
    conf.set("hive.input.format", "org.apache.hadoop.hive.ql.io.CombineHiveInputFormat")  # 执行Map前进行小文件合并
    # map输出合并--没起作用
    conf.set("hive.merge.mapfiles", "true")  # 在Map-only的任务结束时合并小文件
    conf.set("hive.merge.mapredfiles", "true")  # 在Map-Reduce的任务结束时合并小文件
    conf.set("hive.merge.size.per.task", "256*1000*1000")  # 合并文件的大小
    conf.set("hive.merge.smallfiles.avgsize", "16000000")  # 当输出文件的平均大小小于该值时，启动一个独立的map-reduce任务进行文件merge
    conf.set("spark.sql.shuffle.partitions", "800")  # 设置shuffle分区数
    conf.set("spark.driver.maxResultSize", "5g")
    conf.set("spark.sql.hive.mergeFiles", "true")
    # spark=SparkSession(sc)
    # conf.set("spark.kryoserializer.buffer.max","2048M")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    # spark.sql("set hive.exec.dynamic.partition.mode=nonstrict;set hive.exec.dynamic.partition=true;")
    # spark.sql("set hive.exec.max.dynamic.partitions=100000;set hive.exec.max.dynamic.partitions.pernode=100000;set hive.auto.convert.join=true")
    return spark


spark = createSparkSession('test')

# test little files
raw_data = spark.createDataFrame([['a', '2018-08-01', 3],
                                  ['a', '2018-08-02', 4],
                                  ['a', '2018-08-03', 5]], ['sku', 'dt', 'sale'])
spark.sql(''' drop table if exists app.ipc_bca_little_files_lgb_temp ''')
raw_data = raw_data.coalesce(1)
raw_data.registerTempTable('_temp_main_sp')
# spark.sql(''' create table app.ipc_bca_little_files_lgb_temp as select * from _temp_main_sp ''')
spark.sql(''' create table app.ipc_bca_little_files_lgb_temp (
 sku string, 
 dt string,
 sale double
 ) stored as orc''')
spark.sql(''' insert into app.ipc_bca_little_files_lgb_temp select * from _temp_main_sp ''')
# raw_data.write.mode('overwrite').saveAsTable('app.ipc_bca_little_files_lgb_temp')
# raw_data.write.mode("append").insertInto("app.ipc_bca_little_files_lgb_temp", overwrite=True)
# spark.sql(''' insert into app.ipc_bca_little_files_lgb_temp select * from _temp_main_sp ''')
print(raw_data.rdd.getNumPartitions())


### ========================================================
# 计算 kpi 的结果   08-04

conf = {
    'main': {
        'cal_date': '2018-09-13',  # 2018-08-03 start 2018-08-10 end
        'cal_method': 'week',  # day | week
        'cal_args': {
            'day_index': 2,  # index from 0，表示第三天的预测的 mape
            'day_len': 1,
        }},
    'real_data': {
        'sql':  # concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1'), case when length(sale_date) > 10 then concat('_', substring(sale_date, 12)) else '' end)  as sku_id,
            '''
             select 
                 concat(split(sku_code, '/')[0], '_', coalesce(store_id, '-1'))  as sku_id,
                 sale,
                 sale_date as dt_a
             from 
                 app.app_saas_sfs_model_input
             where
                 tenant_id=28 and dt = '{dt}'
            ''',
        # concat(sku_code, '_', coalesce(store_id, '-1'))  as sku_id,
        'table': 'app.app_saas_sfs_model_input',
    },
    'pre_data': {
        # 'sku_id': 'sku_id',
        # 'sku_id': '''concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1')) ''',
        'sku_id': '''concat(sku_code, '_', coalesce(store_id, '-1')) ''',
        # 'table': 'app.app_lgb_test_bca_forecast_result_all_try',
        # 'app.app_saas_sfs_rst' | 'app.app_lgb_test_bca_forecast_result' | app_lgb_test_bca_forecast_result_all_try | app_lgb_test_bca_forecast_result_try_ts
        # | app_zhjs_test_bca_forecast_result_all_try_weekunion | app.app_zhjs_test_bca_forecast_result_all_try_weekunion07
        'table': 'app.app_saas_sfs_rst',
        # 'models': ['wma', 'hw', 'reg_single', 'reg_sum', 'ma'],
        # __none__ | reg_single | hw | wma | combine  | ['wma', 'hw', 'reg_single', 'reg_sum', 'ma']
        'models': ['__none__'],  # __none__ | reg_single | hw | wma | combine | reg_sum
    },
    'kpi': {
        'gap': 5,
        'mape': 0.5,
    },
}


def get_date_list(start_date=None, day_len=None):
    date_list = map(lambda x: x.strftime('%Y-%m-%d'),
                    list(rrule(freq=DAILY, count=day_len, dtstart=parse(start_date))))
    return date_list


def get_real_data(start_date=None, end_date=None, conf=conf):
    # Old method :
    # sp = org_sp.where(''' dt >= '{start_date}' and dt <= '{end_date}' '''.
    #                   format(start_date=start_date, end_date=end_date)). \
    #     groupBy(['sku_id', 'dt']).agg(F.sum(F.col('sale')).alias('real_sale'))
    org_sql = conf['real_data']['sql']
    org_sp = spark.sql(org_sql)
    print('\n\nstart_date : {0}\nend_date : {1}'.format(start_date, end_date))
    # sp = org_sp.where(''' dt >= '{start_date}' and dt <= '{end_date}' '''.
    #                   format(start_date=start_date, end_date=end_date)). \
    #     groupBy(['sku_id', 'dt']).agg(F.sum(F.col('sale')).alias('real_sale'))
    sp = org_sp.where(''' dt_a >= '{start_date}' and dt_a <= '{end_date}' '''.
                      format(start_date=start_date, end_date=end_date)). \
        groupBy(['sku_id']).agg(F.sum(F.col('sale')).alias('real_sale'))
    return sp


def get_pre_data(sfs_date=None, date_list=None, index_list=None, conf=conf):
    sku_id = conf['pre_data']['sku_id']
    table = conf['pre_data']['table']
    models = conf['pre_data']['models']
    pre_sp_list = []
    sfs_date = '2018-08-28'
    for model in models:
        model_str = '' if model == '__none__' else " and sale_type = '{0}' ".format(model)
        sp_list = []
        for i, pre_date in enumerate(date_list):
            pre_sql = '''
                select 
                    {sku_id} as sku_id,
                    sum(cast(split(substring(sale_list, 2, length(sale_list)-2),",")[{i}] as double)) as pre_sale
                from
                    {table}
                where 
                    tenant_id=28 and dt='{sfs_date}' {model_str}
                group by 
                    {sku_id}
            '''.format(sfs_date=sfs_date, i=index_list[i], sku_id=sku_id, table=table, model_str=model_str)
            # tenant_id=28 and dt='{sfs_date}' {model_str}
            if i == 0:
                print(pre_sql)
            sp = spark.sql(pre_sql)
            sp = sp.withColumn('dt_a', F.lit(pre_date))
            sp_list.append(sp)
        res_sp = reduce(lambda x, y: x.union(y), sp_list)
        # 11721364-5/8_K54C
        # new
        res_sp = res_sp.groupBy('sku_id').agg(F.sum(F.col('pre_sale')).alias('pre_sale'))
        pre_sp_list.append(res_sp)
    return pre_sp_list


def get_join_mape(real_sp=None, pre_sp=None):
    # Get all the data
    # all_data_sp = real_sp.select('sku_id').union(pre_sp.select('sku_id')).distinct()
    # gap_sp = all_data_sp.join(real_sp, on=['sku_id'], how='left'). \
    #     join(pre_sp, on=['sku_id'], how='left').na.fill(0)
    #
    # real_sp = real_sp.rdd.toDF(real_sp.columns)
    # pre_sp = pre_sp.rdd.toDF(pre_sp.columns)
    #
    # gap_sp = pre_sp.join(real_sp, on=['sku_id'], how='inner').na.fill(0)
    # gap_sp = pre_sp.join(real_sp, on=['sku_id'], how='right').na.fill(0)
    gap_sp = pre_sp.join(real_sp, on=['sku_id'], how='left').na.fill(0)
    # gap_sp = pre_sp.join(real_sp, on=['sku_id', 'dt'], how='inner')
    # gap_sp = pre_sp.join(real_sp, on=['sku_id'], how='inner')
    return gap_sp


def sku_split(sku_id):
    return '_'.join(sku_id.split('_')[:2])


def cal_kpi(cal_date=None, cal_method=None, conf=conf):
    # cal_date, cal_method, conf=conf['main']['cal_date'], conf['main']['cal_method'], conf
    if cal_method == 'week':
        day_len = 7
        day_index = 0
    else:
        day_index = conf['main']['cal_args']['day_index']
        day_len = conf['main']['cal_args']['day_len']
    # cal the day index
    end_date = cal_date
    start_date = (parse(end_date) - datetime.timedelta(day_len - 1)).strftime('%Y-%m-%d')
    sfs_date = (parse(start_date) - datetime.timedelta(day_index + 1)).strftime('%Y-%m-%d')
    date_list = get_date_list(start_date=start_date, day_len=day_len)
    index_list = range(day_index, day_index + day_len)
    input_table = conf['real_data']['table']
    pre_table = conf['pre_data']['table']
    models = conf['pre_data']['models']
    print('\nstart_date : {0}\nend_date : {1}\nsfs_date : {2}\ndate_list : {3}\nindex_list : {4}\ninput_table : {5}\n'
          'pre_table : {6}\n'.format(start_date, end_date, sfs_date, date_list, index_list, input_table, pre_table))
    # calculate the mape
    real_sp = get_real_data(start_date=start_date, end_date=end_date, conf=conf)
    pre_sp_list = get_pre_data(sfs_date=sfs_date, date_list=date_list, index_list=index_list, conf=conf)
    gap_sp_list = []
    gap_sp_list2 = []
    for i, pre_sp in enumerate(pre_sp_list):
        model = models[i]
        # pre_sp2 = pre_sp.select(F.udf(sku_split)(F.col('sku_id')).alias('sku_id'), 'pre_sale', 'dt') # old
        pre_sp2 = pre_sp.select(F.udf(sku_split)(F.col('sku_id')).alias('sku_id'), 'pre_sale')
        gap_sp = get_join_mape(real_sp=real_sp, pre_sp=pre_sp2)
        gap_sp_list.append(gap_sp)
        gap_sp_list2.append(gap_sp.withColumnRenamed('pre_sale', model + '_sale'))
    # gap_sp = reduce(lambda x, y: x.join(y, on=['sku_id', 'dt', 'real_sale']), gap_sp_list2)
    gap_sp = reduce(lambda x, y: x.join(y, on=['sku_id', 'real_sale']), gap_sp_list2)
    return gap_sp_list, gap_sp


def cal_mape(gap_sp=None, conf=conf):
    models = conf['pre_data']['models']
    gap_columns = gap_sp.columns
    # sum_cond = [F.sum(F.col(x)).alias(x) for x in list(set(gap_columns) - set(['sku_id', 'dt']))] # old
    sum_cond = [F.sum(F.col(x)).alias(x) for x in list(set(gap_columns) - set(['sku_id']))]
    # * (2.0 / 3)
    gap_cond = ['*'] + ['abs(real_sale  - {0}_sale) as {0}_gap'.format(x) for x in models] + \
               ['(real_sale - {0}_sale) as {0}_rel_gap'.format(x) for x in models]
    gap_sp2 = gap_sp.groupBy('sku_id').agg(*sum_cond).selectExpr(*gap_cond)
    agg_cond = [F.sum(F.col(y)).alias(y) for y in ['real_sale'] + map(lambda x: x + '_gap', models)]
    mape_cond = ['({0}_gap / real_sale) as {0}_mape'.format(x) for x in models]
    mape_sp = gap_sp2.select(*agg_cond).selectExpr(*mape_cond)
    return gap_sp2, mape_sp


def cal_mape_d2(gap_sp=None, conf=conf):
    models = conf['pre_data']['models']
    gap_columns = gap_sp.columns
    # sum_cond = [F.sum(F.col(x)).alias(x) for x in list(set(gap_columns) - set(['sku_id', 'dt']))] # old
    sum_cond = [F.sum(F.col(x)).alias(x) for x in list(set(gap_columns) - set(['sku_id']))]
    # * (2.0 / 3)
    gap_cond = ['*'] + ['abs(real_sale  - {0}_sale / 2) as {0}_gap'.format(x) for x in models] + \
               ['(real_sale - {0}_sale) as {0}_rel_gap'.format(x) for x in models]
    gap_sp2 = gap_sp.groupBy('sku_id').agg(*sum_cond).selectExpr(*gap_cond)
    agg_cond = [F.sum(F.col(y)).alias(y) for y in ['real_sale'] + map(lambda x: x + '_gap', models)]
    mape_cond = ['({0}_gap / real_sale) as {0}_mape'.format(x) for x in models]
    mape_sp = gap_sp2.select(*agg_cond).selectExpr(*mape_cond)
    return gap_sp2, mape_sp


def sku_filter(sku_id):
    sku_list = ['XSTD', 'FSTD']
    res_value = 1 if any(map(lambda x: x in sku_id, sku_list)) else 0
    return res_value


# conf['real_data']['sql'] = conf['real_data']['sql'].format(dt='2018-08-13')
conf['real_data']['sql'] = conf['real_data']['sql'].format(dt='ACTIVE')
gap_sp_list, gap_sp = cal_kpi(cal_date=conf['main']['cal_date'], cal_method=conf['main']['cal_method'], conf=conf)
gap_sp.persist()
gap_sp.show()
gap_sp2, mape_sp = cal_mape(gap_sp=gap_sp, conf=conf)
mape_sp.show()
gap_sp2, mape_sp = cal_mape_d2(gap_sp=gap_sp, conf=conf)
mape_sp.show()

# gap_sp2.select('sku_id', 'real_sale', 'reg_single_gap').coalesce(1).write.csv('/user/mart_bca/longguangbin/gap_sp2_csv', mode='overwrite')

# / 2
# 4 - 20641
# +------------------+-----------------+------------------+------------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+------------------+------------------+
# |1.2334488903751435|1.297406426703958|1.0169647169696603|1.0096910135587038|0.9972456983036421|
# +------------------+-----------------+------------------+------------------+------------------+
# 5 - 14333
# +------------------+-----------------+------------------+------------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+------------------+------------------+
# |1.1720888179507842|1.221004717039609|0.9603272515961072|0.9506811256430916|0.9958191284943905|
# +------------------+-----------------+------------------+------------------+------------------+
# 6 - 10329
# +------------------+-----------------+------------------+------------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+------------------+------------------+
# |1.1239744471047821|1.158306445718606|0.8956865855963708|0.8752276258584837|0.9953878142524097|
# +------------------+-----------------+------------------+------------------+------------------+
# 7 - 7638
# +------------------+------------------+------------------+------------------+------------------+
# |          wma_mape|           hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+------------------+------------------+------------------+------------------+
# |1.0900513154525668|1.1137926737933692|0.8573045480732303|0.9534541063884264|0.9950518499551919|
# +------------------+------------------+------------------+------------------+------------------+
# 9 - 4478
# +------------------+-----------------+------------------+-----------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|     reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+-----------------+------------------+
# |1.0469679130434786|1.058932461538462|0.8108711304347829|0.936437725752509|0.9930836120401337|
# +------------------+-----------------+------------------+-----------------+------------------+
# 10 - 3540
# +------------------+------------------+------------------+------------------+-----------------+
# |          wma_mape|           hw_mape|   reg_single_mape|      reg_sum_mape|          ma_mape|
# +------------------+------------------+------------------+------------------+-----------------+
# |1.0314996979569673|1.0421080737120754|0.7780938641145728|0.7793817314871376|0.991905925473427|
# +------------------+------------------+------------------+------------------+-----------------+
# 13 - 1979
# +------------------+----------------+------------------+-----------------+-----------------+
# |          wma_mape|         hw_mape|   reg_single_mape|     reg_sum_mape|          ma_mape|
# +------------------+----------------+------------------+-----------------+-----------------+
# |1.0061014545196536|1.01060586774514|0.7160779409677874|0.724736274301121|0.988888888888889|
# +------------------+----------------+------------------+-----------------+-----------------+
# 15 - 1400
# +------------------+------------------+------------------+------------------+-----------------+
# |          wma_mape|           hw_mape|   reg_single_mape|      reg_sum_mape|          ma_mape|
# +------------------+------------------+------------------+------------------+-----------------+
# |0.9969654363636352|0.9992984218181815|0.7207495963636366|0.7261242618181817|0.988054545454546|
# +------------------+------------------+------------------+------------------+-----------------+

# no
# 4
# +------------------+------------------+-----------------+------------------+------------------+
# |          wma_mape|           hw_mape|  reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+------------------+-----------------+------------------+------------------+
# |1.4836016781175894|1.6089105247157522|1.308854940110659|1.3027981637988695|1.0005350519851641|
# +------------------+------------------+-----------------+------------------+------------------+
# 5
# +------------------+-----------------+------------------+------------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+------------------+------------------+
# |1.3594212979607012|1.455530353932933|1.2111377735077171|1.1433283332300253|0.9976507779086345|
# +------------------+-----------------+------------------+------------------+------------------+
# 6
# +-----------------+------------------+-----------------+------------------+------------------+
# |         wma_mape|           hw_mape|  reg_single_mape|      reg_sum_mape|           ma_mape|
# +-----------------+------------------+-----------------+------------------+------------------+
# |1.261982893327452|1.3283727931447298|1.126731510301808|1.1067875559196014|0.9961943166782182|
# +-----------------+------------------+-----------------+------------------+------------------+
# 7
# +------------------+------------------+------------------+-----------------+------------------+
# |          wma_mape|           hw_mape|   reg_single_mape|     reg_sum_mape|           ma_mape|
# +------------------+------------------+------------------+-----------------+------------------+
# |1.1956587157732361|1.2419656575395288|1.0198431996400565|0.941590326520119|0.9956678236277154|
# +------------------+------------------+------------------+-----------------+------------------+
# 9
# +------------------+-----------------+------------------+-----------------+------------------+
# |          wma_mape|          hw_mape|   reg_single_mape|     reg_sum_mape|           ma_mape|
# +------------------+-----------------+------------------+-----------------+------------------+
# |1.1042158260869568|1.127194153846153|0.8679894448160539|0.884874073578595|0.9907826086956523|
# +------------------+-----------------+------------------+-----------------+------------------+
# 10
# +-----------------+-----------------+------------------+-----------------+------------------+
# |         wma_mape|          hw_mape|   reg_single_mape|     reg_sum_mape|           ma_mape|
# +-----------------+-----------------+------------------+-----------------+------------------+
# |1.073830888162376|1.093799271216455|0.8172398310856835|0.764298848930663|0.9885506061844435|
# +-----------------+-----------------+------------------+-----------------+------------------+
# 13
# +------------------+------------------+------------------+------------------+-----------------+
# |          wma_mape|           hw_mape|   reg_single_mape|      reg_sum_mape|          ma_mape|
# +------------------+------------------+------------------+------------------+-----------------+
# |1.0202130552007942|1.0275770682559957|0.7323322690506596|0.6717178870441325|0.981538243224067|
# +------------------+------------------+------------------+------------------+-----------------+
# 15
# +------------------+------------------+------------------+------------------+------------------+
# |          wma_mape|           hw_mape|   reg_single_mape|      reg_sum_mape|           ma_mape|
# +------------------+------------------+------------------+------------------+------------------+
# |1.0000990472727274|1.0039984727272728|0.6519841672727268|0.6248867636363639|0.9795127272727269|
# +------------------+------------------+------------------+------------------+------------------+


def zhangjs_cal_loss():
    from dateutil.parser import parse
    import datetime
    import pandas as pd
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
        median_absolute_error, r2_score
    from pyspark.sql.types import ArrayType, StringType


    def get_date_range(date_start, pre_len):
        """
        Get the date range list.

        :param str date_start: date start
        :param int pre_len: day length from date start
        :return: list date_list

        Example
        -------
        >>> df = get_date_range('2018-07-03', 3)
        >>> df
        ['2018-07-03', '2018-07-04', '2018-07-05']
        """
        date_start_dt = parse(date_start)
        return [(date_start_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range(pre_len)]


    def split_pre_data_day(pre_data, date_list):
        """
        The additional function of split_pre_data.
        """
        pre_data = eval(pre_data[0]) if isinstance(pre_data[0], (str, unicode)) else pre_data[0]
        date_list = eval(date_list) if isinstance(date_list, (str, unicode)) else date_list
        res_list = [str(pre_data[i]) + '|' + str(date_list[i]) for i in range(len(date_list))]
        return res_list


    def split_pre_data(pre_sp, date_list, cols):
        """
        Split the pre list into pre cols.

        :param sp_df pre_sp: pre sp_df
        :param list date_list: pre date list
        :param list cols: [sku, dt, pre], sku, pre is col_name of pre sp_df, dt is the split_into dt name.
        :return: pre sp_df with a pre col

        Example
        -------
        >>> pre_sp.show()
        | sku     | pre   |
        | 'sku_a' | [2,3] |
        >>> split_pre_data(pre_sp, ['2018-07-03', '2018-07-04'], ['sku', 'dt', 'pre']).show()
        | sku     |      dt      | pre |
        | 'sku_a' | '2018-07-03' | 2   |
        | 'sku_a' | '2018-07-04' | 3   |
        """
        sku, dt, pre = cols[0], cols[1], cols[2]
        split_sp = pre_sp.groupBy(sku).agg(
            F.udf(split_pre_data_day, ArrayType(StringType()))(F.collect_list(F.col(pre)), F.lit(str(date_list))).alias(
                'tmp_split'))
        split_sp = split_sp.withColumn('pre_dt', F.explode(F.col('tmp_split')))
        split_sp = split_sp.withColumn(pre, F.split(F.col('pre_dt'), '\|')[0]). \
            withColumn(dt, F.split(F.col('pre_dt'), '\|')[1]).select(sku, dt, pre)
        return split_sp


    def agg_cal_loss(x, agg_method='day', loss_method='mean_squared'):
        """
        Agg or day level to calculate the loss of predict values.

        :param x: data of a key (dt + real + pre)
        :param agg_method: day | agg
        :param loss_method: explained_variance | mean_absolute | mean_squared | median_absolute | r2_score
        :return: loss value
        """
        loss_methods = {'explained_variance': explained_variance_score,
                        'mean_absolute': mean_absolute_error,
                        'mean_squared': mean_squared_error,
                        'median_absolute': median_absolute_error,
                        'r2_score': r2_score,
                        }
        loss_method = loss_methods.get(loss_method)
        x_df = pd.DataFrame(map(lambda y: list(y), x))
        x_df_columns = x_df.columns
        x_df = x_df.sort_values(x_df_columns[0])
        real_list = map(float, x_df.iloc[:, 1].values)
        pre_list = map(float, x_df.iloc[:, 2].values)
        if agg_method == 'day':
            loss_error = float(loss_method(real_list, pre_list))
        elif agg_method == 'agg':
            loss_error = float(loss_method([sum(real_list)], [sum(pre_list)]))
        else:
            raise Exception(''' agg_method error, your : {0} '''.format(agg_method))
        return loss_error


    def loss(self, period, agg_method, loss_method):
        end_date = self.raw_data.select(F.date_format(F.max(self.ds), self.ds_format)).rdd.map(lambda x: list(x)).first()[0]
        end_date_be = (parse(end_date) - datetime.timedelta(period)).strftime('%Y-%m-%d')
        calculate_loss = self.raw_data.filter(F.col(self.ds) < end_date_be)
        loss_result_rdd = calculate_loss.map(lambda x: ((x[0], x[1]), (x[2], x[3]))).groupByKey(). \
            map(lambda x: _run(x, period))
        pre_loss_sp = loss_result_rdd.toDF().toDF(self.key, 'ts_type', self.target)
        check_loss_sp = self.raw_data.where(''' {0} >= '{1}' and {0} < '{2}' '''.format('dt', end_date_be, end_date))

        date_list = get_date_range(end_date_be, period)
        date_sp = spark.createDataFrame(map(lambda x: [x], date_list), ['dt'])

        main_sp = pre_loss_sp.select('sku').drop_duplicates().crossJoin(date_sp)
        pre_sp = split_pre_data(pre_loss_sp, date_list, ['sku', 'dt', 'pre'])
        main_sp = main_sp.join(check_loss_sp, on=['sku', 'dt'], how='left').join(pre_sp, on=['sku', 'dt'],
                                                                                 how='left').na.fill(0)

        main_sp = main_sp.join(pre_sp, on=['sku']).join(check_loss_sp, on=['sku', 'dt'], how='left')

        loss_sp = main_sp.groupBy('sku').agg(
            F.udf(agg_cal_loss)(F.collect_list(F.struct(F.col('dt'), F.col('sale'), F.col('pre'))), F.lit(agg_method),
                                F.lit(loss_method)).alias('loss'))
        return loss_sp


    def sample_data_run():
        raw_data = spark.createDataFrame([['a', '2018-08-01', 3],
                                          ['a', '2018-08-02', 4],
                                          ['a', '2018-08-03', 5]], ['sku', 'dt', 'sale'])
        pre_loss_sp = spark.createDataFrame([['a', [2, 3, 3, 5]]], ['sku', 'pre'])

        end_date = '2018-08-04'
        period = 3
        method = 'day'
        loss_method = 'mean_squared'

        end_date_be = (parse(end_date) - datetime.timedelta(period)).strftime('%Y-%m-%d')
        check_loss_sp = raw_data.where(''' {0} >= '{1}' and {0} < '{2}' '''.format('dt', end_date_be, end_date))

        date_list = get_date_range(end_date_be, period)
        date_sp = spark.createDataFrame(map(lambda x: [x], date_list), ['dt'])

        main_sp = pre_loss_sp.select('sku').drop_duplicates().crossJoin(date_sp)
        pre_sp = split_pre_data(pre_loss_sp, date_list)
        main_sp = main_sp.join(check_loss_sp, on=['sku', 'dt'], how='left').join(pre_sp, on=['sku', 'dt'],
                                                                                 how='left').na.fill(0)

        main_sp.persist()
        main_sp.show()

        main_sp.groupBy('sku').agg(
            F.udf(agg_cal_loss)(F.collect_list(F.struct(F.col('dt'), F.col('sale'), F.col('pre'))), F.lit(method),
                                F.lit(loss_method)).alias('loss')).show()
        loss_sp = main_sp.groupBy('sku').agg(
            F.udf(agg_cal_loss)(F.collect_list(F.struct(F.col('dt'), F.col('sale'), F.col('pre'))), F.lit(method),
                                F.lit(loss_method)).alias('loss'))





# 0.9548099140910437

def get_all_date(date_begin, date_end):
    date_begin_dt = parse(date_begin)
    date_end_dt = parse(date_end)
    date_len = (date_end_dt - date_begin_dt).days + 1
    return [(date_begin_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range(date_len)]


import pprint
from copy import deepcopy

date_list = get_all_date('2018-08-13', '2018-08-27')
res_list = []
for dt_i in date_list:
    conf2 = deepcopy(conf)
    conf2['real_data']['sql'] = conf['real_data']['sql'].format(dt=dt_i)
    pprint.pprint(conf2)
    gap_sp_list, gap_sp = cal_kpi(cal_date=conf['main']['cal_date'], cal_method=conf['main']['cal_method'], conf=conf2)
    gap_sp2, mape_sp = cal_mape(gap_sp=gap_sp, conf=conf)
    tmp = mape_sp.toPandas()
    tmp2 = tmp.copy()
    res_list.append([dt_i, tmp2])

pprint.pprint(res_list)

mm = gap_sp.select('sku_id', 'real_sale', 'ma_sale').groupBy('sku_id').agg(F.sum(F.col('real_sale')).alias('real_sale'),
                                                                           F.sum(F.col('ma_sale')).alias('ma_sale')). \
    selectExpr('*', 'abs(real_sale - ma_sale) as ma_gap')
mm.select(F.sum(F.col('real_sale')).alias('real_sale'),
          F.sum(F.col('ma_gap')).alias('ma_gap')).show()

gap_sp2.where(''' sku_id not like '%XSTD%' and sku_id not like '%FSTD%' ''').select('sku_id',
                                                                                    'reg_single_rel_gap').orderBy(
    'reg_single_rel_gap', ascending=False).show()
gap_sp2.select('sku_id', 'reg_single_rel_gap').orderBy('reg_single_rel_gap', ascending=False).show()

gap_sp2.where(''' sku_id not like '%XSTD%' and sku_id not like '%FSTD%' '''). \
    select('sku_id', 'real_sale', 'reg_single_gap'). \
    select(F.sum('real_sale').alias('real_sale'), F.sum('reg_single_gap').alias('reg_single_gap')).show()

# +---------+-----------------+
# |real_sale|   reg_single_gap|
# +---------+-----------------+
# |   5037.0|7487.278900001175|
# +---------+-----------------+

gap_sp2.select('sku_id', 'real_sale', 'reg_single_gap'). \
    select(F.sum('real_sale').alias('real_sale'), F.sum('reg_single_gap').alias('reg_single_gap')).show()

s = spark.sql(''' show create table app.app_lgb_test_bca_forecast_result_all_try ''')
ss = list(s.collect()[0])[0]

cal_date = conf['main']['cal_date']
day_len = 7
day_index = 0
end_date = cal_date
start_date = (parse(end_date) - datetime.timedelta(day_len - 1)).strftime('%Y-%m-%d')
real_sp = get_real_data(start_date=start_date, end_date=end_date, conf=conf)
real_sp.select(F.sum('real_sale')).show()
real_sp.groupBy('sku_id').agg(F.count('*').alias('cnt')).orderBy('cnt', ascending=False).show()
gap_sp.select(F.sum('real_sale')).show()

# +---------+------------------+
# |real_sale|    reg_single_gap|
# +---------+------------------+
# |  18290.0|13422.559933337328|
# +---------+------------------+

gap_sp2.where(''' sku_id not like '%XSTD%' and sku_id not like '%FSTD%' '''). \
    select('sku_id', 'real_sale', 'reg_single_gap')

gap_sp2.select('sku_id', 'real_sale', 'reg_single_gap').coalesce(1).write.csv('/user/mart_bca/longguangbin/gap_sp2_csv')

'reg_single_rel_gap'

# 08-04 : 0.9321122252309649

filter_sp = gap_sp.withColumn('flag', F.udf(sku_filter)(F.col('sku_id')))
filter_sp2, mape_sp2 = cal_mape(filter_sp.where(''' flag = 0 '''), conf)
# mape_sp.show()
mape_sp2.show()

gap_sp2.persist()
gap_sp2.show()

# check 'real_sale', 'reg_single_sale', 'wma_sale', 'combine_sale'
gap_sp.select(F.sum('real_sale').alias('real_sale'), F.sum('reg_single_sale').alias('reg_single_sale'),
              F.sum('wma_sale').alias('wma_sale'), F.sum('combine_sale').alias('combine_sale')).show()

# +------------------+------------------+------------------+
# |          wma_mape|           hw_mape|      combine_mape|
# +------------------+------------------+------------------+
# |0.8084083645888879|0.7746974241431138|0.7840974420035081|
# +------------------+------------------+------------------+


sp1 = spark.table('app.lgb_delete_tt_1')
sp2 = spark.table('app.lgb_delete_tt_2')

spark.sql('''
select sum(sales.qty) as qty
    ,sum(forecast.predict) as predict
    ,count(distinct sales.item) as sku_cnt
    ,sum(abs(coalesce(sales.qty, 0) - coalesce(forecast.predict, 0))) as absolute_error
    ,sum(abs(coalesce(sales.qty, 0) - coalesce(forecast.predict, 0))) / sum(coalesce(sales.qty, 0)) as mape
from
    app.lgb_delete_tt_1  sales
right join
    app.lgb_delete_tt_2  forecast
on
    sales.item = forecast.item
    and sales.store = forecast.store
''').show()

spark.sql('''select 
    sum(a.qty) as qty
    ,sum(a.predict) as predict
    ,count(distinct a.item) as sku_cnt
    ,sum(abs(a.qty - a.predict)) as absolute_error
    ,sum(abs(a.qty - a.predict)) / sum(a.qty) as mape
from 
    (   
        select 
            sales.item as item,
            coalesce(sales.qty, 0) as qty,
            coalesce(forecast.predict, 0) as predict
        from
            app.lgb_delete_tt_1  sales
        right join
            app.lgb_delete_tt_2  forecast
        on
            sales.item = forecast.item
            and sales.store = forecast.store
    ) a''').show()

mm = spark.sql('''select 
            forecast.item as item,
            sales.qty as qty,
            forecast.predict as predict
        from
            app.lgb_delete_tt_1  sales
        right join
            app.lgb_delete_tt_2  forecast
        on
            sales.item = forecast.item
            and sales.store = forecast.store''')
mm.where(''' qty is null or predict is null ''').show()

"""
with sales as
    (select store,item,sum(sale) as qty 
    from 
      (select sku_code as item,store_id as store, sale 
      from app.app_saas_sfs_model_input 
      where tenant_id = 28 and dt = 'ACTIVE' and sale_date >='2018-08-01' and sale_date <= '2018-08-07'
      ) t 
      group by item,store
    ),
forecast as
    (
    select
        store
        ,item
        ,coalesce(sum(sale_list_2), 0) as predict
    from 
        (
        select
            *
        from
            (
            select
               sku_code as item
               ,store_id as store
               ,sale_list
            from
               app.app_zhjs_test_bca_forecast_result_all_try_ts
            where
               sale_type='reg_single'
               and tenant_id=28 and dt='2018-08-19'
            ) a lateral view explode(split(split(split(sale_list, '\\[')[1], '\\]')[0], ',')) b as sale_list_2
        ) a
    group by store,item
    )
select sum(sales.qty) as qty
    ,sum(forecast.predict) as predict
    ,count(distinct sales.item) as sku_cnt
    ,sum(abs(sales.qty - forecast.predict)) as absolute_error
    ,sum(abs(sales.qty - forecast.predict)) / sum(sales.qty) as mape
from
    sales
join
    forecast
on
    sales.item = forecast.item
    and sales.store = forecast.store
"""


def sku_filter_more(sku_id):
    sku_list = ['15829177-1', '19827252-2', '15821185-3']
    res_value = 1 if any(map(lambda x: x in sku_id, sku_list)) else 0
    return res_value


def get_min_gap(l):
    return min(list(l))


models = conf['pre_data']['models']
min_gap = gap_sp2.withColumn('min_gap', F.udf(get_min_gap)(F.struct([F.col(x + '_gap') for x in models])))
min_gap.select(F.sum('min_gap').alias('gap'), F.sum('real_sale').alias('real_sale')). \
    withColumn('mape', F.col('gap') / F.col('real_sale')).show()

# 分析思路：现在做了一些调整（后处理方面的）。
# 1、将一些销量低的做拦截
# 2、将近期仅有少数销量的，做（60）45天（一个半月）填充，求出 quantile

# 问题 sku ： XSTD(塑料袋 棕色) | FSTD(塑料袋 棕色)
# 08-04
# +----------------+------------------+------------------+------------------+
# | reg_single_mape|           hw_mape|          wma_mape|      combine_mape|
# +----------------+------------------+------------------+------------------+
# |0.94985343179391|1.0480319827563753|1.3455046899012792|0.9991541533171843|
# +----------------+------------------+------------------+------------------+
# base_day：45
# 均销量高：((sale_sum / 有销量天数) > 4 )
# 数据稀疏：((有销量天数 / base_day) < 0.25 )


### ========================================================
# 计算 kpi 的结果

# 07-30 - dev    - all_gap : 115193.23010000012, all_real : 78248.0, mape : 1.472155583529293, kpi: 0.999320397987
# 07-30 - online - all_gap : 136006.36717854845, all_real : 78248.0, mape : 1.738144964453384, kpi: 0.999320397987
# 07-30 - dev2   - all_gap : 87135.89179999994,  all_real : 78248.0, mape : 1.113586184950413, kpi: 0.998347796795

# 07-31 - online - all_gap : 144380.381021159,   all_real : 74407.0, mape : 1.940413953272662, kpi: 0.999549514119
# 07-31 - dev2   - all_gap : 113220.840200000,   all_real : 74407.0, mape : 1.521642321286978, kpi: 0.999510841734

# 08-01 - online - all_gap : 144355.74508572236, all_real : 74141.0, mape : 1.947043404940887, kpi: 0.999557616839
# 08-02 - online - all_gap : 145181.2996679934,  all_real : 74553.0, mape : 1.947356909420055, kpi: 0.999616822179

# 分析：1、以 08-01 + 08-02 分析为主，因为模型修正。（因为想做一个 ensemble 的实验，看看有效果没有）


### ========================================================
# 尝试去覆盖今天的任务

online_sql = '''
select 
    sku_code, store_id, channel_id,
    concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1')) as sku_id,
    dynamic_dims, sale_type, pre_target_dimension_id
from
    app.app_saas_sfs_rst 
where 
    dt = '{dt}'
'''

dev_sql = '''
select 
    sku_id, sale_list, std_list
from
    app.app_lgb_test_bca_forecast_result
where 
    dt = '{dt}'
'''


def get_join_online(online_dt, dev_dt):
    online_sp = spark.sql(online_sql.format(dt=online_dt))
    dev_sp = spark.sql(dev_sql.format(dt=dev_dt))
    join_sp = online_sp.join(dev_sp, on=['sku_id'])
    print('online_sp : {0}'.format(online_sp.count()))  # 181846
    print('dev_sp : {0}'.format(dev_sp.count()))  # 181846
    print('join_sp : {0}'.format(join_sp.count()))  # 181846
    join_sp = join_sp.select(
        *['sku_code', 'store_id', 'channel_id', 'dynamic_dims', 'sale_type', 'sale_list', 'std_list',
          'pre_target_dimension_id'])
    return join_sp


def insert_online(sp, dt):
    sp.registerTempTable("_sparkTemp_partition_sp")
    table = 'app.app_saas_sfs_rst'
    overwrite_str = 'overwrite'
    partition_str = ''' tenant_id='28', dt='{dt}' '''.format(dt=dt)
    spark.sql(Template(
        """ insert ${overwrite} table ${table} partition(${partition_str}) select * from _sparkTemp_partition_sp """).substitute(
        table=table, partition_str=partition_str, overwrite=overwrite_str))


# online_dt = '2018-08-12'
online_dt = 'ACTIVE'
dev_dt = '2018-08-12'

# join_sp = get_join_online(online_dt=online_dt, dev_dt=dev_dt)
# join_sp.write.saveAsTable('app.app_lgb_tt_combine_try', mode='overwrite')
# insert_online(sp=join_sp, dt=online_dt)

### ========================================================
# ensemble 后的覆盖结果
online_sale_sql = '''
select 
    sku_code, store_id, channel_id,
    concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1')) as sku_id,
    dynamic_dims, sale_type, pre_target_dimension_id, sale_list, std_list
from
    app.app_saas_sfs_rst 
where 
    dt = '{dt}'
'''


def get_weekend(start_date, end_date):
    this_day = parse(start_date).weekday()
    start_date = parse(start_date)
    end_date = parse(end_date)
    sat_delta = 6 if this_day > 5 else 5 - this_day
    sun_delta = 6 - this_day
    sat_day = start_date + datetime.timedelta(sat_delta)
    sun_day = start_date + datetime.timedelta(sun_delta)
    day_list = []
    while (sat_day < end_date) or (sun_day < end_date):
        if sat_day < end_date:
            day_list.append(sat_day)
            sat_day = sat_day + datetime.timedelta(7)
        if sun_day < end_date:
            day_list.append(sun_day)
            sun_day = sun_day + datetime.timedelta(7)
    week_list = sorted(map(lambda x: x.strftime('%Y-%m-%d'), day_list))
    return week_list


def get_date_range(start_date, end_date):
    start_date_dt = parse(start_date)
    end_date_dt = parse(end_date)
    date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime("%Y-%m-%d"),
                     range((end_date_dt - start_date_dt).days + 1))
    return date_range


def get_week_df(start_date, end_date):
    week_list = get_weekend(start_date, end_date)
    date_list = get_date_range(start_date, end_date)
    date_pd = pd.DataFrame(date_list, columns=['dt'])
    week_pd = pd.DataFrame(week_list, columns=['dt'])
    week_pd['week'] = '1'
    week_df = date_pd.merge(week_pd, on=['dt'], how='left').fillna('0')
    return week_df


def combine_res(l, week_list):
    l_list = list(l)
    on_sale_list = eval(l_list[0])
    sale_list = eval(l_list[1])
    week_list = eval(week_list)
    res = []
    for i, each in enumerate(week_list):
        if each == '1':
            res.append((float(on_sale_list[i]) + float(sale_list[i])) / 2)
        else:
            res.append(sale_list[i])
    # raise Exception(res)
    return str(res)


def get_ensemble_sp(sfs_date, start_date, end_date):
    ## test ensemble method
    online_sp = spark.sql(online_sale_sql.format(dt=sfs_date))
    dev_sp = spark.sql(dev_sql.format(dt=sfs_date))
    online_basic_sp = online_sp.select('sku_code', 'store_id', 'channel_id', 'sku_id', 'dynamic_dims', 'sale_type',
                                       'pre_target_dimension_id')
    online_need_sp = online_sp.select('sku_id', 'sale_list', 'std_list').withColumnRenamed('sale_list', 'on_sale_list'). \
        withColumnRenamed('std_list', 'on_std_list')
    # get week data
    week_df = get_week_df(start_date, end_date)
    week_df['index'] = range(len(week_df))
    week_list = week_df['week'].values.tolist()
    # join data
    join_sp = online_need_sp.join(dev_sp, on=['sku_id'])
    join_sp = join_sp.select('sku_id', 'on_sale_list', 'sale_list')
    # ensemble data
    ensemble_sp = join_sp.withColumn('combine', F.udf(combine_res, StringType())(
        F.struct(F.col('on_sale_list'), F.col('sale_list')), F.lit(str(week_list))))
    res_sp = ensemble_sp.select('sku_id', 'combine').withColumnRenamed('combine', 'sale_list'). \
        withColumn('std_list', F.col('sale_list'))
    all_sp = res_sp.join(online_basic_sp, on=['sku_id']).select(
        *['sku_code', 'store_id', 'channel_id', 'dynamic_dims', 'sale_type', 'sale_list', 'std_list',
          'pre_target_dimension_id'])
    # get the result
    dev_count = dev_sp.count()
    online_count = online_sp.count()
    all_count = all_sp.count()
    all_sp.show()
    print('\ndev_count : {0}\nonline_count : {1}\nall_count : {2}\n'.format(dev_count, online_count, all_count))
    return all_sp


sfs_date = '2018-08-09'
start_date = '2018-08-10'
end_date = (parse(start_date) + datetime.timedelta(90)).strftime('%Y-%m-%d')

all_sp = get_ensemble_sp(sfs_date, start_date, end_date)

all_sp.write.saveAsTable('app.app_lgb_tt_combine_try', mode='overwrite')

# insert_online(all_sp, dt='2018-08-09')
# insert_online(all_sp, dt='ACTIVE')
