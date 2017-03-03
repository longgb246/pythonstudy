#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os

path = r'D:\Lgb\Workspace'
read_path = path + os.sep + 'table_sample_sku.csv'
inv_data = pd.read_csv(read_path)

inv_data_sort = inv_data.sort_values(['fdc', 'sku_x', 'date_s'])
inv_data_sort.to_csv(path + os.sep + 'inv_data_sort.csv', index=False)


from collections import defaultdict
import multiprocessing


class test:
    def __init__(self):
        self.a = [1,2,3,4,5,6,7]
        self.fdc = list('abc')
        self.qtty = defaultdict(lambda: defaultdict(int))
        self.pool = multiprocessing.Pool(processes=4)

    def cal(self, x):
        if x > 3:
            for f in self.fdc:
                self.qtty[f + str(x)][x] = x
        else:
            self.qtty[str(x)][1] = x

    def mul(self):
        self.result = self.pool.map(self.cal, self.a)

a = test()
a.mul()
print a.result



import cPickle as pickle
path = r'D:\Lgb\Workspace'
fdc_alt = path + os.sep + 'fdc_alt.pkl'
fdc_alt_prob = path + os.sep + 'fdc_alt_prob.pkl'
fdc_alt = pickle.load(open(fdc_alt))
fdc_alt_prob = pickle.load(open(fdc_alt_prob))
fdc_alt_pd = pd.DataFrame([fdc_alt['628'], fdc_alt_prob['628']]).T
fdc_alt_pd.columns = ['alt','p']
fdc_alt_pd.sort_values(['p'])

fdc_alt_pd_no = fdc_alt_pd[:34]

import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.bar(fdc_alt_pd_no['alt'] , fdc_alt_pd_no['p'], color="#0070C0",  align='center')
ax.set_xlim = [9,40]




path = 'D:\Lgb'
sample_path = path + os.sep + 'table_sample_sku.csv'
data_sample = pd.read_csv(sample_path)
data_sample = data_sample.sort_values(['fdc','sku_x','date_s'])
data_sample.columns
select_list = [3046109, 3133827, 1152819, 695438, 605077, 312721, 2341748, 312721, 1027498,
               3319158, 1080065, 605077, 3023380 ,1038240 ,1861124 ,1578164 ,109750]
select_list = list(set(select_list))



select = map(lambda x: x in select_list ,data_sample['sku_x'].values)

data_3473576 = data_sample[data_sample['sku_x'] == 3473576]
data_3473576.to_csv(path + os.sep + 'sale_3473576.csv', index=False)



from collections import Counter
Counter(select)
data_select = data_sample[select]
data_drop_dup = data_select.drop_duplicates()
data_drop_dup.to_csv(path + os.sep + 'select_sku_dup.csv', index=False)


data_838689 = data_sample[data_sample['sku_x']==838689].drop_duplicates()
data_852061 = data_sample[data_sample['sku_x']==852061].drop_duplicates()




data_sample[(data_sample['sku_x']==1038240) & (data_sample['date_s']=='2016-10-02')].drop_duplicates()




sku_path = path + os.sep + '2016-10-02.pkl'
data_sku = pickle.load(open(sku_path))




path = r'D:\Lgb'
allocation_sale_data = path + os.sep + 'aa_allocation_sale_data.csv'
sample = path + os.sep + 'aa_sample.csv'
tmp_allocation_sale_data =  path + os.sep + 'aa_tmp_allocation_sale_data.csv'

allocation_sale_data = pd.read_csv(allocation_sale_data)
sample = pd.read_csv(sample)
tmp_allocation_sale_data = pd.read_csv(tmp_allocation_sale_data)



a = sample.merge(allocation_sale_data, left_on=['sku'], right_on=['item_sku_id'])
a = a.drop_duplicates()
a.to_csv(path + os.sep + 'sale.csv', index=False)


skus = pd.read_table(path + os.sep + 'skus.txt', header=None)
skus.columns = ['sku']
skus['sku'][0]  = skus['sku'][0][3:]
a['sku'] = a['sku'].astype(str)
skus['sku'] = skus['sku'].astype(str)
a_keep = a.merge(skus, on=['sku'])
a_keep.to_csv(path + os.sep + 'sale_actual1_sim1.csv', index=False)
