#-*- coding:utf-8 -*-
#===============================================================================
#
#         FILE: extractText.py
#
#  DESCRIPTION: 用于合并文本与全信息数据
#      OPTIONS: ---
# REQUIREMENTS: ---
#        NOTES: ---
#       AUTHOR: lgb453476610@163.com
#      COMPANY: self
#      VERSION: 1.0
#      CREATED: 2017-09-04
#===============================================================================
from __future__ import division
import os
import pandas as pd
import json


def transforVar(df, column_name):
    '''
    用于将变量的值进行编码，变成string的:'1','2','3',...
    '''
    # df, column_name = df, column_name
    high_limit = 20
    values_name = list(df[column_name].drop_duplicates())
    try:
        values_name.remove('other')
    except:
        pass
    int_str = map(lambda x: str(x), range(1, high_limit))
    categories = {'other':'0'}
    categories_write = {'other':'0'}
    for i, each in enumerate(values_name):
        each_str = each.encode('utf-8') if type(each) == unicode else str(each)
        categories[each] = int_str[i]
        categories_write[each_str] = int_str[i]
    df[column_name] = df.loc[:,[column_name]].applymap(categories.get)
    return df, categories_write


def transforVarCols(df, column_names):
    '''
    用于将变量的值进行编码，变成string的:'1','2','3',...
    '''
    # df, column_names = attr_file, attr_this
    categories_list = []
    for column_name in column_names:
        # column_name = column_names[1]
        df, categories = transforVar(df, column_name)
        categories_list.append([column_name, categories])
    return df, categories_list


read_path = r'D:\Work\Codes\inventory_optimization\personal\longguangbin\Test\xxxata\datasets'
save_path = r'D:\Work\Codes\inventory_optimization\personal\longguangbin\Test\xxxata\datasets\combine'
info_file_name = 'cleaned_first_sales_price_df.csv'
attr_files = 'sku_basic_info_{0}.csv'
cate_list = [655, 672, 866, 867, 1049]
attr_list = [['kadai', 'phone_type', 'color', 'communication_G', 'memory_GB', 'hard_GB'],
             ['color', 'size', 'iDai', 'Monitor', 'memory_GB', 'hard_GB'],
             ['brand', 'color', 'shell', 'proThrow'],
             ['steel', 'glass', 'crack', 'highDefine', 'blue', '3D', 'color', 'brand'],
             ['color', 'length']]


if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    info_file = pd.read_csv(read_path + os.sep + info_file_name, encoding='gbk')
    categories = []
    for i, cate in enumerate(cate_list):
        # i, cate = 0, cate_list[0]
        print cate
        # 1、读取全信息文件
        info_file_this = info_file[info_file['item_third_cate_cd']==cate]
        attr_this = attr_list[i]
        # 2、读取特征信息文件
        attr_file_name = attr_files.format(cate)
        attr_file = pd.read_csv(read_path + os.sep + attr_file_name, encoding='gbk')
        attr_file = attr_file.fillna('other')
        # 3、变量编码
        attr_file_code, categories_this = transforVarCols(attr_file, attr_this)
        categories.append([cate, categories_this])
        # 4、合并文件
        info_file_combine = info_file_this.merge(attr_file_code[['item_sku_id_hashed']+attr_this], on=['item_sku_id_hashed'])
        # 5、输出文件
        info_file_combine.to_csv(save_path + os.sep + 'sku_basic_info_combine_{0}.csv'.format(cate), index=False, encoding='gbk')
    # 输出编码文件，json文件用于解析
    with open(save_path + os.sep + 'categories.json', 'w') as f:
        json.dump(categories, f, ensure_ascii=False)
    # txt文件，用于查看
    with open(save_path + os.sep + 'categories.txt', 'w') as f:
        for i, each in enumerate(categories):
            f.write(str(each[0]))
            f.write('\n')
            attr_writes = each[1]
            for attr_write in attr_writes:
                f.write('\t')
                f.write(str(attr_write[0]))
                f.write('\t\n')
                comma_list = sorted(attr_write[1].iteritems(), key=lambda x: int(x[1]))
                for each_comma in comma_list:
                    f.write('\t\t')
                    f.write(str(each_comma[0]))
                    f.write(' : ')
                    f.write(each_comma[1])
                    f.write('\n')
                f.write('\n')
            f.write('\n')



