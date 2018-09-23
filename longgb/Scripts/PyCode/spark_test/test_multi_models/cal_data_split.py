# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/5
  Usage   : 
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window

spark = SparkSession.builder.enableHiveSupport().getOrCreate()


def get_split_sample():
    """ Get the sample data from anta_offline to split into band. """

    get_sample_sql = '''
        select 
            concat(sku_code, '_', coalesce(store_id, '-1'), '_', coalesce(channel_id, '-1'), case when length(sale_date) > 10 then concat('_', substring(sale_date, 12)) else '' end) as sku_id,
            sale, 
            case when length(sale_date) > 10 then substring(sale_date, 1, 10) else sale_date end as dt 
        from 
            app.app_saas_sfs_model_input
        where 
            tenant_id = 28
            and dt = '2018-08-04'
            and sku_code in ('11721101-1/10.5', '11721101-1/6.5', '11721101-1/7', '11721101-1/7.5', '11721101-1/8',  
            '11721101-1/9.5', '11721101-3/10.5', '11721101-3/7', '11721101-3/8', '11721101-3/8.5', '11721101-3/9.5', 
            '11721303-1/6.5', '11721303-1/7', '11721303-1/7.5', '11721303-1/8', '11721303-1/8.5', '11721303-1/9', 
            '11721303-10/10.5', '11721303-10/6.5', '11721303-10/7', '11721303-10/7.5', '11721303-10/8', '11721303-10/8.5', 
            '11721303-10/9.5', '11721303-3/10.5', '11721303-3/6.5', '11721303-3/7', '11721303-3/7.5', '11721303-3/8', 
            '11721303-3/9', '11721306-3/10.5', '11721306-3/7', '11721306-3/7.5', '11721306-3/8', '11721306-3/8.5',
            '11721360-5/10.5', '11721360-5/7', '11721360-5/8', '11721360-5/8.5', '11721360-5/9.5', '11721360-6/9.5', 
            '11721101-1/9', '11721303-1/10.5', '11721303-1/9.5', '11721303-10/9', '11721303-3/8.5', '11721360-7/10.5', 
            '11721360-7/7', '11721360-7/8', '11721101-1/8.5')
        '''
    sample_sp = spark.sql(get_sample_sql)
    return sample_sp


def cal_sku_band(sp):
    """ Calculate the band of sp. """

    sku_id = 'sku_id'
    sale = 'sale'

    split_list = [0.2, 0.4, 0.6, 0.8]
    split_name = [1, 2, 3, 4, 5]
    cond_str = ''
    for i, percent in enumerate(split_list):
        cond_str += '''        WHEN rank_percent < {percent} THEN {name} '''.format(percent=percent, name=split_name[i])
    cond_str += '''        ELSE {name}'''.format(name=split_name[-1])
    split_cond = '''
    CASE 
        {cond_str}
    END 
        AS band
    '''.format(cond_str=cond_str)

    group_sp = sp.groupBy(sku_id).agg(F.sum(F.col(sale)).alias('sale_sum'))
    distinct_sp = group_sp.select(F.countDistinct(sku_id).alias('cnt'))
    windowspec_r = Window.orderBy(F.col('sale_sum').desc())
    rank_sp = group_sp.select('sku_id', F.rank().over(windowspec_r).alias('rank')).crossJoin(distinct_sp)
    rank_sp = rank_sp.withColumn('rank_percent', F.col('rank') / F.col('sale_sum'))
    band_sp = rank_sp.selectExpr('sku_id', split_cond)

    return band_sp


def cal_data_split(sp=None, save_path=''):
    """ Calculate the split measure of data. """

    # get sample split data.
    sp = get_split_sample() if sp is None else sp
    band_sp = cal_sku_band(sp)
    band_sp.show()
    band_sp.coalesce(1).write.csv(save_path)  # '/user/mart_bca/longguangbin/sku_tmp_band_csv'


if __name__ == '__main__':
    cal_data_split()
