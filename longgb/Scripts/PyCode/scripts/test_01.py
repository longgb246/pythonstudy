# -*- coding:utf-8 -*-
import os

if __name__ == '__main__':
    path = os.getcwd()
    sku_id = '10001'
    file = path + os.sep + 'test.txt'
    f = open(file, 'w')
    for i in range(1, 10):
        f.write(sku_id)
        f.write('\t')
        f.write('2017-01-0{0}'.format(i))
        f.write('\t')
        f.write('{0}'.format(i * 10))
        f.write('\n')
    f.close()

import pandas as pd

pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', 150)  # 150

train = pd.read_csv(r'/Users/longguangbin/Downloads/train_20181125105600.csv', sep='|')
flag = pd.read_csv(r'/Users/longguangbin/Downloads/flag_20181125105020.csv', sep='|')
granu = pd.read_csv(r'/Users/longguangbin/Downloads/granu_split_20181125175310.csv', sep='|')

granu['sku_id_s'] = granu['sku_id'].apply(lambda x: x.split('$')[1])
granu_s = granu.loc[:, ['sku_id_s', 'granu_split']].drop_duplicates()
granu_s.index = range(len(granu_s))
granu_s = granu_s.groupby(['sku_id_s']).count().reset_index().rename(columns={'granu_split': 'cnt'})
granu_m = granu.merge(granu_s, on=['sku_id_s'])
check_df = granu_m[granu_m['cnt'] == 2]
check_df.index = range(len(check_df))
check_df.to_csv(r'/Users/longguangbin/Downloads/check_1.csv', index=False)

flag = flag.drop_duplicates()
flag.index = range(len(flag))

flag_2 = flag.copy()
flag_2.columns = ['a', 'b']
flag_2['b'] = 'this is a long test' + flag_2['b']

train.drop_duplicates()
