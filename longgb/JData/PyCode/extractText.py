#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from collections import Counter
import jieba
import warnings
warnings.filterwarnings('ignore')
import re


read_path = r'D:\Data\other\JData\2-JData_train'
save_path = r'D:\Data\other\JData\Analysis'
file_name = 'train_sku_basic_info.csv'


if __name__ == '__main__':
    basic = pd.read_table(read_path + os.sep + file_name, encoding='gbk')
    cate3 = list(basic['item_third_cate_cd'].drop_duplicates().values)

    # # 【这里是分词操作】
    # for cate in cate3:
    #     # cate = cate3[0]
    #     print cate
    #     basic_tmp = basic[basic['item_third_cate_cd']==cate]
    #     basic_tmp['sku_name'] = map(lambda x: x.replace(' ',''),basic_tmp['sku_name'].values)
    #     all_words = reduce(lambda x, y: x+y, map(lambda x: jieba.lcut(x), basic_tmp['sku_name'].values))
    #     all_words_count = pd.DataFrame.from_dict(Counter(all_words), orient='index').reset_index().rename(columns={0:'count', 'index':'words'}).sort_values(['count'], ascending=False)
    #     all_words_count.index = range(len(all_words_count))
    #     # all_words_count.to_csv(save_path + os.sep + 'all_words_count_{0}.csv'.format(cate), index=False, encoding='gbk')
    #     all_words_count.to_csv(save_path + os.sep + 'all_words_count_jin_{0}.csv'.format(cate), index=False, encoding='gbk')
    #     # 进行词贴紧


    # 【这里按照空格进行分割】
    cate = cate3[0]
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    words_space = reduce(lambda x, y: x+y, map(lambda x: x.split(' '), basic_tmp['sku_name'].values))
    all_words_count = pd.DataFrame.from_dict(Counter(words_space), orient='index').reset_index().rename(columns={0:'count', 'index':'words'}).sort_values(['count'], ascending=False)
    all_words_count.index = range(len(all_words_count))

    all_words_count.to_csv(save_path + os.sep + 'all_words_count_space_{0}.csv'.format(cate), index=False, encoding='gbk')

    text_analysis = basic_tmp.loc[:,['sku_name']]
    text_analysis.index = range(len(text_analysis))


    kadai = u'双卡双待'
    count_kadai = map(lambda x: 1 if kadai in x else 0, text_analysis['sku_name'].values)
    text_analysis['kadai'] = count_kadai
    text_analysis['sku_name'] = map(lambda x: x.replace(kadai, ''), text_analysis['sku_name'].values)

    Counter(count_kadai)

    x = text_analysis['sku_name'][0]


    def splitPhone(x):
        phone_key = [u'全网通', u'移动', u'联通', u'电信']
        key_str = u''
        for key in phone_key:
            if key in x and key not in key_str:
                key_str = key_str + u' ' + key
        if np.min(map(lambda x: x in key_str, phone_key)) or np.min(map(lambda x: x in key_str, phone_key[1:])):
            key_str = u' 全网通'
        return key_str

    text_analysis['phone_type'] = map(lambda x: splitPhone(x), text_analysis['sku_name'].values)


    def splitColor(x):
        color_list = [u'粉', u'X色', u'金', u'银', u'铜', u'红', u'橙', u'黄', u'绿', u'蓝', u'青', u'紫', u'靛', u'黑', u'褐', u'白', u'灰']
        # x_list = jieba.lcut(x)
        x_list = x.split(' ')
        drop_list = [u'黄金', u'金立', u'金属', u'金刚', u'金钢', u'金典', u'流金', u'红辣椒', u'红米', u'青玉', u'青春', u'文青', u'青岛', u'青海', u'青漾', u'魅蓝', u'蓝牙', u'蓝魔', u'黑龙江']
        i = 0
        color_str_list = []
        for each in x_list:
            if np.min(map(lambda x: x not in each, drop_list)):
                for color in color_list:
                    if color in each and color not in color_str_list:
                        i += 1
                        color_str_list.append(color)
        # 多种颜色处理
        if u'黑' in color_str_list:
            color_str = u'黑'
        elif u'白' in color_str_list:
            color_str = u'白'
        elif len(color_str_list) >= 2:
            color_str = color_str_list[-1]
        elif len(color_str_list) == 1:
            color_str = color_str_list[0]
        else:
            color_str = u''

        # if color_str == u'':
        #     return True
        # else:
        #     return False
        return color_str



# :  2203
# 白  :  1990
# 金  :  1703
# 黑  :  1499
# 银  :  814
# 灰  :  813
# 蓝  :  359
# 红  :  216
# 黄  :  117
# 绿  :  104
# 紫  :  39
# 青  :  36
# 橙  :  31
# 褐  :  3



    mm = text_analysis.loc[map(lambda x: splitColor(x), text_analysis['sku_name'].values),:]
    mm.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')


    aa = map(lambda x: splitColor(x), text_analysis['sku_name'].values)


    for key, v in sorted(Counter(aa).items(), key=lambda x: x[1], reverse=True):
        print key, ' : ', v



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



