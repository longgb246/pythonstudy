#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os


hive_sql = '''
select
    dc_id,
    sku_id,
    sum(total_inv)as inv ,
    sum(total_sales) as sales,
    sum(total_inv)/sum(total_sales) as ito
from (
    SELECT
        A.dc_id    ,
        A.date_s   ,
        a.sku_id,
        A.total_inv,
        B.total_sales
    FROM
        dev.dev_fdc_daily_total_inv_sku A
    JOIN
        dev.dev_fdc_daily_sales_inv_sku B
    ON
        A.date_s    = B.date_s
        AND A.dc_id = B.dc_id
        and a.sku_id=b.item_sku_id
    ) a
group by
    dc_id,
    sku_id
'''


if __name__ == '__main__':
    os.system('hive -e "{0}"  >  sku_ito; '.format(hive_sql))

