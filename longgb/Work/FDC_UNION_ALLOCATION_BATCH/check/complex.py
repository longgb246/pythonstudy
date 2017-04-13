#-*- coding:utf-8 -*-
import os
import pandas as pd
from string import Template


def getDateRange(start_date, end_date, freq='D'):
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date, freq=freq).values)
    return date_range


def hive_sql(sql_str, file_name):
    os.system('''hive -e "{0}" > {1}.out; '''.format(sql_str, file_name))


if __name__=="__main__":
    sql = '''
select
    a.item_sku_id,
    a.dt1,
    b.dt2
from
    (
        select
            dt as dt1,
            item_sku_id
        from
            gdm.gdm_m03_item_sku_da
        where
            dt = '$this_date'
    )  a
left join
    (
        select
            dt as dt2,
            item_sku_id
        from
            gdm.gdm_m03_item_sku_da
        where
            dt = '$next_date'
    )  b
on
    a.item_sku_id = b.item_sku_id
where
    b.dt2 is null
    '''
    sql = Template(sql)
    start_date = '2017-01-01'
    end_date = '2017-04-13'
    date_range = getDateRange(start_date, end_date)
    for i, each_date in enumerate(date_range):
        print each_date
        hive_sql(sql.substitute(this_date=each_date, next_date=date_range[i+1]), each_date)

