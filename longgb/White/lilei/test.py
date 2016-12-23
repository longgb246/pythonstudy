dtype = {'parent_sale_ord_id': str}
df = pd.read_csv('D:/white_list/white_list_history'+str(i)+'.csv', sep='\t', dtype=dtype)

import os
import pandas as pd
path  = r'D:\white_list2'


def dataclean(x):
    x = str(x)
    tmp_x = x.split('\002')
    return str(tmp_x)

for i in [i for i in range(1,4)]:
    data = pd.read_csv(path + os.sep + 'white_list_history'+str(1)+'.csv')
    data['list_item_sku'] = map(dataclean, data['list_item_sku'].values)
    data.head(8)
    data.to_csv('D:/white_list/white_list_history' +str(3)+ str(1) + '.csv',index=False, sep='\t')


data = pd.read_csv(path + os.sep + 'white_list_history'+str(1)+'.csv')
data['list_item_sku'] = map(dataclean, data['list_item_sku'].values)
data.head

