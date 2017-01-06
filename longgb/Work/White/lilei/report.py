import numpy as np
import pandas as pd
import datetime
import time

dtype = {'parent_sale_ord_id': str}
df = pd.read_csv('D:/white_list/white_list_sales_5days.csv', sep='\t', dtype=dtype)(df)
test_date_set = time.strptime('2016 - 10 - 01', '%Y - %m - %d')
test_date_next = test_date_set - datetime.timedelta( days  = 3 )
test_date =test_date_set[i];    #测试日期

history_start_date= test_date -datetime.timedelta
history_end_date

dtype = {'parent_sale_ord_id': str}


wl_old = pd.read_csv('', sep='\t', dtype=dtype)
df = pd.read_csv('E:/data/white_list_src_data.csv', sep='\t', dtype=dtype)
parent_order_id = df['parent_sale_ord_id']
order_sku = df['list_item_sku']