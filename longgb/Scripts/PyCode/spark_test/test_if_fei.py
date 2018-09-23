#-*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/1
  Usage   : 
"""    


# print sql
d_sum = 'sum(cast(split(substring(sale_list, 2, length(sale_list)-1),",")[{i}] as double)) as dsum_{i}'
d_sum_list = []
for i in range(91):
    d_sum_list.append(d_sum.format(i=i))
d_sum_str = ',\n    '.join(d_sum_list)


sum_sql = '''
select 
    {d_sum_str}
from 
    app.dev_lgb_test_saas_sfs_rst 
where 
    dt = '2018-07-30'
'''.format(d_sum_str=d_sum_str)

print(sum_sql)



# print sql2
d_sum = 'sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[{i}] as double)) as dsum_{i}'
d_sum_list = []
for i in range(91):
    d_sum_list.append(d_sum.format(i=i))
d_sum_str = ',\n    '.join(d_sum_list)
sum_sql = '''
select
    {d_sum_str}
from 
    (
        select 
            sku_code, store_id, channel_id,
            min(dynamic_dims) as dynamic_dims,
            min(sale_type) as sale_type,
            min(sale_list) as sale_list,
            min(std_list) as std_list,
            min(pre_target_dimension_id) as pre_target_dimension_id,
            min(tenant_id) as tenant_id,
            min(dt) as dt    
        from 
            app.app_saas_sfs_rst 
        where 
            dt='ACTIVE'
        group by 
            sku_code, store_id, channel_id   
    )   a
'''.format(d_sum_str=d_sum_str)
print(sum_sql)



import matplotlib.pyplot as plt
import pandas as pd

path = r'/Users/longguangbin/Downloads/234_app_11759427.xls'
path2 = r'/Users/longguangbin/Downloads/234_app_11759875.xls'

data = pd.read_excel(path)
data2 = pd.read_excel(path2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.values.tolist()[0][:-1])
ax.set_yticks([0, 25000])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data2.values.tolist()[0][:-1])
ax.set_yticks([0, 25000])
plt.show()



# Get the sale sum
sale_list = []
for each in range(90):
    sale_list.append('round(sum(cast(split(substring(a.sale_list, 2, length(a.sale_list)-1),",")[{0}] as double)), 2)'.format(each))
sale_str = ',\n                '.join(sale_list)
sale_sql = '''
select 
    demand_sum 
from 
    (
        select
            array(
                {sale_str}
            ) as demand
        from 
            (
                select 
                    sale_list
                from 
                    app.app_saas_sfs_rst 
                where 
                    dt = date_add(CURRENT_DATE, -1)
            )   a
    )   b
lateral view explode(b.demand) 
      c as demand_sum
'''.format(sale_str=sale_str)
print(sale_sql)



