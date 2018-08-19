# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@jd.com
  Date    : 2018/8/8
  Usage   : 
"""

#### jdvl start -m /data0/mart_bca:/data0/mart_bca:rw -i bdp-docker.jd.com:5000/wise_mart_bca:latest -o='--net=host' -I bash
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_cib:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_cib:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/jdk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/jdk1.8.0_121/jre/lib/amd64/server

from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY
import datetime
from string import Template
import pandas as pd
import pprint

from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

### ========================================================
# 计算 kpi 的结果

conf = {
    'main': {
        'cal_date': '2018-08-10',
        'cal_method': 'week',  # day | week
        'cal_args': {
            'day_index': 2,  # index from 0，表示第三天的预测的 mape
            'day_len': 1,
        }},
    'real_data': {
        'sql':  # concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1'), case when length(sale_date) > 10 then concat('_', substring(sale_date, 12)) else '' end)  as sku_id,
            '''
             select 
                 concat(sku_code, '_', coalesce(store_id, '-1'))  as sku_id,
                 sale,
                 sale_date as dt
             from 
                 app.app_saas_sfs_model_input
             where
                 tenant_id=28 and dt = 'ACTIVE'
            ''',
        'table': 'app.app_saas_sfs_model_input',
    },
    'pre_data': {
        'sku_id': 'sku_id',
        # 'sku_id': '''concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1')) ''',
        'table': 'app.app_lgb_test_bca_forecast_result_all_try',
        # 'app.app_saas_sfs_rst' | 'app.app_lgb_test_bca_forecast_result' | app_lgb_test_bca_forecast_result_try | app_lgb_test_bca_forecast_result_try_ts
        'models': ['reg_single', 'hw', 'wma', 'combine'],  # __none__ | reg_single | hw | wma | combine
    },
    'kpi': {
        'gap': 5,
        'mape': 0.5,
    },
}


def get_date_list(start_date, day_len):
    date_list = map(lambda x: x.strftime('%Y-%m-%d'),
                    list(rrule(freq=DAILY, count=day_len, dtstart=parse(start_date))))
    return date_list


def get_real_data(start_date, end_date, conf=conf):
    org_sql = conf['real_data']['sql']
    org_sp = spark.sql(org_sql)
    print('\n\nstart_date : {0}\nend_date : {1}'.format(start_date, end_date))
    sp = org_sp.where(''' dt >= '{start_date}' and dt <= '{end_date}' '''.
                      format(start_date=start_date, end_date=end_date)). \
        groupBy(['sku_id', 'dt']).agg(F.sum(F.col('sale')).alias('real_sale'))
    return sp


def get_pre_data(sfs_date, date_list, index_list, conf=conf):
    sku_id = conf['pre_data']['sku_id']
    table = conf['pre_data']['table']
    models = conf['pre_data']['models']
    pre_sp_list = []
    for model in models:
        model_str = '' if model == '__none__' else " and sale_type = '{0}' ".format(model)
        sp_list = []
        for i, pre_date in enumerate(date_list):
            pre_sql = '''
                select 
                    {sku_id} as sku_id,
                    sum(cast(split(substring(sale_list, 2, length(sale_list)-1),",")[{i}] as double)) as pre_sale
                from
                    {table}
                where 
                    tenant_id=28 and dt='{sfs_date}' {model_str}
                group by 
                    {sku_id}
            '''.format(sfs_date=sfs_date, i=index_list[i], sku_id=sku_id, table=table, model_str=model_str)
            if i == 0:
                print(pre_sql)
            sp = spark.sql(pre_sql)
            sp = sp.withColumn('dt', F.lit(pre_date))
            sp_list.append(sp)
        res_sp = reduce(lambda x, y: x.union(y), sp_list)
        pre_sp_list.append(res_sp)
    return pre_sp_list


def get_join_mape(real_sp, pre_sp):
    # Get all the data
    all_data_sp = real_sp.select('sku_id', 'dt').union(pre_sp.select('sku_id', 'dt')).distinct()
    gap_sp = all_data_sp.join(real_sp, on=['sku_id', 'dt'], how='left'). \
        join(pre_sp, on=['sku_id', 'dt'], how='left').na.fill(0)
    return gap_sp


def sku_split(sku_id):
    return '_'.join(sku_id.split('_')[:2])


def cal_kpi(cal_date, cal_method, conf=conf):
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
    date_list = get_date_list(start_date, day_len)
    index_list = range(day_index, day_index + day_len)
    input_table = conf['real_data']['table']
    pre_table = conf['pre_data']['table']
    models = conf['pre_data']['models']
    print('\nstart_date : {0}\nend_date : {1}\nsfs_date : {2}\ndate_list : {3}\nindex_list : {4}\ninput_table : {5}\n'
          'pre_table : {6}\n'.format(start_date, end_date, sfs_date, date_list, index_list, input_table, pre_table))
    # calculate the mape
    real_sp = get_real_data(start_date, end_date, conf=conf)
    pre_sp_list = get_pre_data(sfs_date, date_list, index_list, conf=conf)
    gap_sp_list = []
    gap_sp_list2 = []
    for i, pre_sp in enumerate(pre_sp_list):
        model = models[i]
        pre_sp2 = pre_sp.select(F.udf(sku_split)(F.col('sku_id')).alias('sku_id'), 'pre_sale', 'dt')
        gap_sp = get_join_mape(real_sp, pre_sp2)
        gap_sp_list.append(gap_sp)
        gap_sp_list2.append(gap_sp.withColumnRenamed('pre_sale', model + '_sale'))
    gap_sp = reduce(lambda x, y: x.join(y, on=['sku_id', 'dt', 'real_sale']), gap_sp_list2)
    return gap_sp_list, gap_sp


def cal_mape(gap_sp, conf):
    models = conf['pre_data']['models']
    gap_columns = gap_sp.columns
    sum_cond = [F.sum(F.col(x)).alias(x) for x in list(set(gap_columns) - set(['sku_id', 'dt']))]
    gap_cond = ['*'] + ['abs(real_sale - {0}_sale) as {0}_gap'.format(x) for x in models] + \
               ['(real_sale - {0}_sale) as {0}_rel_gap'.format(x) for x in models]
    gap_sp2 = gap_sp.groupBy('sku_id').agg(*sum_cond).selectExpr(*gap_cond)
    agg_cond = [F.sum(F.col(y)).alias(y) for y in ['real_sale'] + map(lambda x: x + '_gap', models)]
    mape_cond = ['({0}_gap / real_sale) as {0}_mape'.format(x) for x in models]
    mape_sp = gap_sp2.agg(*agg_cond).selectExpr(*mape_cond)
    return gap_sp2, mape_sp


def sku_filter(sku_id):
    sku_list = ['XSTD', 'FSTD']
    res_value = 1 if any(map(lambda x: x in sku_id, sku_list)) else 0
    return res_value


gap_sp_list, gap_sp = cal_kpi(conf['main']['cal_date'], conf['main']['cal_method'], conf=conf)
gap_sp2, mape_sp = cal_mape(gap_sp, conf)

filter_sp = gap_sp.withColumn('flag', F.udf(sku_filter)(F.col('sku_id')))
filter_sp2, mape_sp2 = cal_mape(filter_sp.where(''' flag = 0 '''), conf)

mape_sp.show()
mape_sp2.show()

gap_sp2.persist()
gap_sp2.show()


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

# 最低：reg - 0.9675153257893531
# 问题 sku ： XSTD(塑料袋 棕色) | FSTD(塑料袋 棕色)
# 以下模型中剔除了塑料袋，计算 mape 时候，分别加上和不加塑料袋
#            加上塑料袋    |     不计算塑料袋
# reg - 1.048386517049641 | 1.0595368663138631
# hw - 1.1791193421317077 | 1.220396196646718
# wma - 1.454485044550142 | 1.5592180836506913
# combine - 1.11819407248 | 1.1454311060553608
# 61752.0 | 64643.97430000002 | 1.046832075074492
# 5827.126400000001

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
