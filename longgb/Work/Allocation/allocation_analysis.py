#-*- coding:utf-8 -*-
import os
import cPickle as pickle
import numpy as np
import pandas as pd
import time
import re


def printruntime(t1, name):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d >0:
        print 'Run Time ({3}) is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print 'Run Time ({2}) is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


def gene_index(fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s)+str(fdc)+str(sku)


def getdaterange(start_date,end_date):
    '''
    生成日期，格式'yyyy-mm-dd'
    '''
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date).values)
    return date_range


def getpkl(pkl_path):
    '''
    读取pkl文件
    '''
    with open(pkl_path) as f:
        data = pickle.load(f)
    return data


def parse_fdc_inv(fdc_inv_func):
    '''
    解析fdc_inv数据
    '''
    fdc_inv_pd = []
    fdc_inv_pd_names = ["date_s", "fdc", "sku", "inv"]
    for k0, v0 in fdc_inv_func.items():
        tmp_fdc_inv_pd = []
        tmp_fdc_inv_pd.append(k0[:10])  # date_s
        tmp_fdc_inv_pd.append(k0[10:13])  # fdc
        tmp_fdc_inv_pd.append(k0[13:])  # sku
        for k1, v1 in v0.items():
            if k1 == 'inv':
                tmp_fdc_inv_pd.append(v1)  # inv
        fdc_inv_pd.append(tmp_fdc_inv_pd)
    fdc_inv_pd = pd.DataFrame(fdc_inv_pd, columns=fdc_inv_pd_names)
    return fdc_inv_pd


def parse_tab_txt(path_data, method=0):
    '''
    解析tab分割文件，
    method，看具体生成文件的脚本而定。
    method = 0 ：表示\n下面生成的文件解析。如下：
    with open(save_data_path+'simu_orders_retail','w') as orl:
        for k,v in allocation.simu_orders_retail.items():
            for k1,v1 in v.items():
                for k2,v2 in v1.items():
                    orl.write(str(k))
                    orl.write('\t')
                    orl.write(str(k1))
                    orl.write('\t')
                    orl.write(str(k2))
                    orl.write('\t')
                    orl.write(str(v2))
            orl.write('\n')
    method = 1 ：表示\n下面的文件解析。如下：
    with open(save_data_path+'simu_orders_retail','w') as orl:
        for k,v in allocation.simu_orders_retail.items():
            for k1,v1 in v.items():
                for k2,v2 in v1.items():
                    orl.write(str(k))
                    orl.write('\t')
                    orl.write(str(k1))
                    orl.write('\t')
                    orl.write(str(k2))
                    orl.write('\t')
                    orl.write(str(v2))
                    orl.write('\n')
    '''
    data_pd = []
    data_name = ['fdc','date_s','sale_ord_tm','sale_ord_id','sku','sale_qtty']
    if method == 1:
        # 修改后，每一个项目加了回车以后的解析方法
        for eachline in open(path_data):
            tmp_data = []
            each = eachline.split('\t')
            if len(each) >= 3:
                tmp_data.append(each[0][:3])        # fdc
                tmp_data.append(each[0][3:])        # date_s
                try:
                    tmp_data.append(each[1][:21])
                except:
                    print 'Error 0'
                    print each[1]
                # tmp_data.append(each[1][:21])       # sale_ord_tm
                tmp_data.append(each[1][21:])       # sale_ord_id
                tmp_data.append(each[2])            # sku
                tmp_data.append(each[3])            # sale_qtty
                data_pd.append(tmp_data)
        data_pd = pd.DataFrame(data_pd, columns=data_name)
    else:
        # 修改前，很多项在一行的解析方法
        for eachline in open(path_data):
            each = eachline.split('\t')
            if len(each) >= 3:
                max_each = len(each)
                max_split = int(np.floor(max_each / 3))
                for i in xrange(max_split):
                    tmp_data = []
                    tmp_start = i * 3
                    if i == 0:
                        tmp_data.append(each[tmp_start][:3])        # fdc
                        tmp_data.append(each[tmp_start][3:])        # date_s
                    else:
                        tmp_split = re.findall('(.*)(630|628|658)(.*)', each[tmp_start])[0]
                        try:
                            tmp_data.append(tmp_split[1])               # fdc
                            tmp_data.append(tmp_split[2])               # date_s
                        except:
                            print 'Error 1'
                            print tmp_split
                    tmp_data.append(each[tmp_start+1][:21])         # sale_ord_tm
                    tmp_data.append(each[tmp_start+1][21:])         # sale_ord_id
                    tmp_data.append(each[tmp_start+2])              # sku
                    if i == (max_split - 1):
                        tmp_data.append(each[tmp_start+3])          # sale_qtty
                    else:
                        tmp_split = re.findall('(.*)(630|628|658)(.*)', each[tmp_start+3])[0]
                        tmp_data.append(tmp_split[0])               # sale_qtty
                    data_pd.append(tmp_data)
        # print len(data_pd)
        # print data_pd[0]
        data_pd = pd.DataFrame(data_pd, columns=data_name)
    return data_pd


# ===========================================================================
# =                                 参数设置                                 =
# ===========================================================================
# 读取文件路径
path_analysis = r'/home/cmo_ipc/Allocation_shell/longgb/allocation_one_month_org'
path_fdc_inv = path_analysis + os.sep + 'fdc_inv'
path_fdc_simu_orders_retail = path_analysis + os.sep + 'fdc_simu_orders_retail'
path_simu_orders_retail = path_analysis + os.sep + 'simu_orders_retail'
# 存储文件路径
save_path = r'/home/cmo_ipc/Allocation_shell/longgb/allocation_one_month_org'


if __name__ == '__main__':
    # ===========================================================================
    # =                             （1）读取并解析数据                          =
    # ===========================================================================
    print '开始读取并解析数据...'
    # 1、读取并解析fdc_inv数据
    t1 = time.time()
    fdc_inv = getpkl(path_fdc_inv)
    printruntime(t1, '读取 fdc_inv 数据')
    t1 = time.time()
    t1 = time.time()
    fdc_inv_pd = parse_fdc_inv(fdc_inv)
    printruntime(t1, '解析 fdc_inv 数据')
    len_fdc_inv_pd = len(fdc_inv_pd.index)
    print '[ Data Anlysis ] fdc_inv Rows is : {0}'.format(len_fdc_inv_pd)

    # 2、读取并解析fdc_simu_orders_retail数据
    t1 = time.time()
    fdc_simu_orders_retail_pd = parse_tab_txt(path_fdc_simu_orders_retail)
    printruntime(t1, '读取并解析 fdc_simu_orders_retail 数据')
    len_fdc_simu_orders_retail_pd = len(fdc_simu_orders_retail_pd.index)
    print '[ Data Anlysis ] fdc_simu_orders_retail Rows is : {0}'.format(len_fdc_simu_orders_retail_pd)

    # 3、读取并解析fdc_inv数据
    t1 = time.time()
    simu_orders_retail_pd = parse_tab_txt(path_simu_orders_retail)
    printruntime(t1, '读取并解析 simu_orders_retail 数据')
    len_simu_orders_retail_pd = len(simu_orders_retail_pd.index)
    print '[ Data Anlysis ] simu_orders_retail Rows is : {0}'.format(len_simu_orders_retail_pd)

    # ===========================================================================
    # =                             （2）生成周转、满足率表                       =
    # ===========================================================================
    t1 = time.time()
    # 4.1、生成ito周转表，含sku
    table_ito_sku = fdc_inv_pd.merge(fdc_simu_orders_retail_pd, on=["date_s","fdc","sku"], how='inner')
    table_ito_sku = table_ito_sku.loc[:,["date_s","fdc","inv","sale_qtty", "sku"]].drop_duplicates()
    table_ito_sku["inv"] = map(float, table_ito_sku["inv"].values)
    table_ito_sku["sale_qtty"] = map(float, table_ito_sku["sale_qtty"].values)
    print table_ito_sku.head()

    # 4.2、生成ito周转表，不含sku
    table_ito = table_ito_sku.groupby(["date_s","fdc","inv"])["sale_qtty"].sum().reset_index()
    table_ito["sale_qtty"] = map(float, table_ito["sale_qtty"].values)
    table_ito["ito_rate"] = table_ito["inv"] / (table_ito["sale_qtty"] + 0.0000001)

    # 5、生成fill满足率表
    tmp_table_fill_01 = fdc_simu_orders_retail_pd.groupby(["fdc", "date_s"])["sale_ord_id"].count().reset_index()
    tmp_table_fill_01.columns = ["fdc", "date_s", "fdc_order_num"]
    tmp_table_fill_02 = simu_orders_retail_pd.groupby(["fdc", "date_s"])["sale_ord_id"].count().reset_index()
    tmp_table_fill_02.columns = ["fdc", "date_s", "total_order_num"]
    table_fill = tmp_table_fill_01.merge(tmp_table_fill_02, on=["fdc", "date_s"], how='inner')
    table_fill["fill_rate"] = table_fill["fdc_order_num"] * 1.0 / table_fill["total_order_num"]
    printruntime(t1, '生成 table_ito_sku、table_ito、table_fill 表')

    # ===========================================================================
    # =                             （3）保存数据表                              =
    # ===========================================================================
    t1 = time.time()
    # 6、保存生成表
    table_ito_sku.to_csv(save_path + os.sep + 'table_ito_sku.csv', index=False)
    table_ito.to_csv(save_path + os.sep + 'table_ito.csv', index=False)
    table_fill.to_csv(save_path + os.sep + 'table_fill.csv', index=False)

    # 7、保存原表
    fdc_inv_pd.to_csv(save_path + os.sep + 'fdc_inv_pd.csv', index=False)
    fdc_simu_orders_retail_pd.to_csv(save_path + os.sep + 'fdc_simu_orders_retail_pd.csv', index=False)
    simu_orders_retail_pd.to_csv(save_path + os.sep + 'simu_orders_retail_pd.csv', index=False)
    printruntime(t1, '生成 table_ito_sku、table_ito、table_fill 表')
    print '[ Data Anlysis ] Finish !'

