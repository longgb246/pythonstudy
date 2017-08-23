#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from collections import Counter
import jieba
import warnings
warnings.filterwarnings('ignore')


read_path = r'D:\Data\other\JData\2-JData_train'
save_path = r'D:\Data\other\JData\Analysis'
file_name = 'train_sku_basic_info.csv'


if __name__ == '__main__':
    basic = pd.read_table(read_path + os.sep + file_name, encoding='gbk')
    cate3 = list(basic['item_third_cate_cd'].drop_duplicates().values)
    for cate in cate3:
        # cate = cate3[0]
        print cate
        basic_tmp = basic[basic['item_third_cate_cd']==cate]
        basic_tmp['sku_name'] = map(lambda x: x.replace(' ',''),basic_tmp['sku_name'].values)
        all_words = reduce(lambda x, y: x+y, map(lambda x: jieba.lcut(x), basic_tmp['sku_name'].values))
        all_words_count = pd.DataFrame.from_dict(Counter(all_words), orient='index').reset_index().rename(columns={0:'count', 'index':'words'}).sort_values(['count'], ascending=False)
        all_words_count.index = range(len(all_words_count))
        # all_words_count.to_csv(save_path + os.sep + 'all_words_count_{0}.csv'.format(cate), index=False, encoding='gbk')
        all_words_count.to_csv(save_path + os.sep + 'all_words_count_jin_{0}.csv'.format(cate), index=False, encoding='gbk')
        # 进行词贴紧




    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    words_space = reduce(lambda x, y: x+y, map(lambda x: x.split(' '), basic_tmp['sku_name'].values))
    all_words_count = pd.DataFrame.from_dict(Counter(words_space), orient='index').reset_index().rename(columns={0:'count', 'index':'words'}).sort_values(['count'], ascending=False)
    all_words_count.index = range(len(all_words_count))
    all_words_count.to_csv(save_path + os.sep + 'all_words_count_space_{0}.csv'.format(cate), index=False, encoding='gbk')



    # 检验：
    # 1、'item_desc' 是个没什么用的字段
    # basic_used_count = Counter(basic['item_desc'])
    # for key, value in basic_used_count.iteritems():
    #     print key, ' : ', value

    # 2、'data_type' 全部是 1

    # 3、统计品牌的信息
    # brand_count = basic.groupby(['barndname_full'])['item_sku_id_hashed'].count().reset_index().rename(columns={'item_sku_id_hashed':'count'}).sort_values(['count'], ascending=False)
    # brand_count.index = range(len(brand_count))
    # brand_all = pd.DataFrame.from_dict(Counter(basic['barndname_full']), orient='index').reset_index().rename(columns={0:'count', 'index':'brand'}).sort_values(['count'], ascending=False)
    # brand_all.index = range(len(brand_all))

    # 4、对于颜色的统计
    # pd.DataFrame.from_dict(Counter(basic['colour']), orient='index').reset_index().rename(columns={0:'count', 'index':'color'}).sort_values('count', ascending=False)


    # app.fdc_predict_forecast_result
    pass

