#-*- coding:utf-8 -*-
import os
import pandas as pd
from string import Template


def getDateRange(start_date, end_date, freq='D'):
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date, freq=freq).values)
    return date_range


if __name__=="__main__":
    sql2 = '''
select
    count(*)
from
    gdm.gdm_m03_item_sku_da
where
    dt = '$this_date'
    '''
    sql2 = Template(sql2)
    start_date = '2017-01-01'
    end_date = '2017-04-13'
    date_range = getDateRange(start_date, end_date)
    for i, each_date in enumerate(date_range):
        print each_date
        os.system('echo -n "{0},   " >> easy_sql.out'.format(each_date))
        os.system('''hive -e "{0}" >> easy_sql.out; '''.format(sql2.substitute(this_date=each_date)))
