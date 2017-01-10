# -*- coding:utf-8 -*-
import os
import cPickle as pickle
import numpy as np
import pandas as pd
import time, datetime
import re

# 分析抽样sku的波动情况:库存，调拨量，销量，预测量

# 读取文件路径
path_analysis = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3'
path_fdc_inv = path_analysis + os.sep + 'fdc_inv.pkl'
path_fdc_allocation = path_analysis + os.sep + 'fdc_allocation.pkl'
path_fdc_simu_orders_retail = path_analysis + os.sep + 'fdc_simu_orders_retail.txt'
path_simu_orders_retail = path_analysis + os.sep + 'simu_orders_retail.txt'
# 存储文件路径
save_path = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/'
# 原始订单数据
sale_data_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sale/'
sku_data_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sku/'
sku_data_alone = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sku/2016-10-02.pkl'


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
    fdc_inv_pd_names = ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po"]
    for k0, v0 in fdc_inv_func.items():
        tmp_fdc_inv_pd = []
        tmp_fdc_inv_pd.append(k0[:10])  # date_s
        tmp_fdc_inv_pd.append(k0[10:13])  # fdc
        tmp_fdc_inv_pd.append(k0[13:])  # sku
        tmp_fdc_inv_pd.append(v0.get('inv', -1))  # inv
        tmp_fdc_inv_pd.append(v0.get('allocation', -1))  # allocation
        tmp_fdc_inv_pd.append(v0.get('arrive_quantity', -1))  # arrive_quantity
        tmp_fdc_inv_pd.append(v0.get('open_po', -1))  # open_po
        # for k1, v1 in v0.items():
        #     if k1 == 'inv':
        #         tmp_fdc_inv_pd.append(v1)  # inv
        #     if k1=='allocation':
        #         tmp_fdc_inv_pd.append(v1)  # allocation
        #     if k1=='arrive_quantity':
        #         tmp_fdc_inv_pd.append(v1)  # arrive_quantity
        fdc_inv_pd.append(tmp_fdc_inv_pd)
    fdc_inv_pd = pd.DataFrame(fdc_inv_pd, columns=fdc_inv_pd_names)
    return fdc_inv_pd


def parse_fdc_allocation(fdc_allocation_func):
    '''
    解析fdc_allocation数据
    '''
    pass


def printruntime(t1, name):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d > 0:
        print 'Run Time ({3}) is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print 'Run Time ({2}) is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


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
    data_name = ['fdc', 'date_s', 'sale_ord_tm', 'sale_ord_id', 'sku', 'sale_qtty']
    if method == 1:
        # 修改后，每一个项目加了回车以后的解析方法
        for eachline in open(path_data):
            tmp_data = []
            each = eachline.split('\t')
            if len(each) >= 3:
                tmp_data.append(each[0][:3])  # fdc
                tmp_data.append(each[0][3:])  # date_s
                try:
                    tmp_data.append(each[1][:21])
                except:
                    print 'Error 0'
                    print each[1]
                # tmp_data.append(each[1][:21])       # sale_ord_tm
                tmp_data.append(each[1][21:])  # sale_ord_id
                tmp_data.append(each[2])  # sku
                tmp_data.append(each[3])  # sale_qtty
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
                        tmp_data.append(each[tmp_start][:3])  # fdc
                        tmp_data.append(each[tmp_start][3:])  # date_s
                    else:
                        tmp_split = re.findall('(.*)(630|628|658)(.*)', each[tmp_start])[0]
                        try:
                            tmp_data.append(tmp_split[1])  # fdc
                            tmp_data.append(tmp_split[2])  # date_s
                        except:
                            print 'Error 1'
                            print tmp_split
                    tmp_data.append(each[tmp_start + 1][:21])  # sale_ord_tm
                    tmp_data.append(each[tmp_start + 1][21:])  # sale_ord_id
                    tmp_data.append(each[tmp_start + 2])  # sku
                    if i == (max_split - 1):
                        tmp_data.append(each[tmp_start + 3].strip('\n'))  # sale_qtty
                    else:
                        tmp_split = re.findall('(.*)(630|628|658)(.*)', each[tmp_start + 3])[0]
                        tmp_data.append(tmp_split[0].strip('\n'))  # sale_qtty
                    data_pd.append(tmp_data)
        # print len(data_pd)
        # print data_pd[0]
        data_pd = pd.DataFrame(data_pd, columns=data_name)
    return data_pd


def datelist(start, end):
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


def to_num_value(L):
    try:
        T = eval(L)
        return sum(T)
    except:
        return -1


if __name__ == '__main__':
    # ===========================================================================
    # =                             （1）读取并解析数据                          =
    # ===========================================================================
    # 随机抽取n个sku
    print r'随机抽取sku...'
    n_sku = 100
    sku_sample_list = []
    with open(save_path + 'white_list_dict_02.txt', 'r') as white:
        for l in white.readlines():
            sku_sample_list.extend(eval(l.split('\t')[2]))
        sku_sample_list = list(set(sku_sample_list))
        sku_sample_list = map(str, sku_sample_list)

    ####抽样SKU用以分析###########
    # sample_sku_list=pd.DataFrame(['1468837'],columns=['sku'])
    sample_sku_list = np.random.choice(sku_sample_list, n_sku)
    sample_sku_list = np.concatenate((sample_sku_list,
                                      np.array(['3107131', '1083590', '779351', '1655954',
                                                '2607943', '2341788', '830487', '2250151', '1968825',
                                                '1231217426', '765776', '1468837', '2352646',
                                                '1000054', '1000274', '1000564', '1944463', '3051749', '326467',
                                                '1944291', '1942991'])))
    sample_sku_list = pd.DataFrame(np.array(sample_sku_list).reshape(-1, 1))
    sample_sku_list.columns = ['sku']
    print '抽样结果'
    print sample_sku_list
    print r'开始读取并解析数据...'
    # 1、读取并解析fdc_inv数据
    t1 = time.time()
    fdc_inv = getpkl(path_fdc_inv)
    printruntime(t1, r'读取 fdc_inv 数据,包括库存，调拨，到达三个数值')
    t1 = time.time()
    fdc_inv_pd = parse_fdc_inv(fdc_inv)
    # print fdc_inv_pd[fdc_inv_pd['sku'] == '3107131']
    printruntime(t1, r'解析 fdc_inv 数据,包括库存，调拨，到达三个数值')
    len_fdc_inv_pd = len(fdc_inv_pd.index)
    print r'[ Data Anlysis ] fdc_inv Rows is : {0}'.format(len_fdc_inv_pd)

    ##2，获取销量数据，包括仿真FDC销量，仿真总销量，实际总销量
    # 读取并解析fdc_simu_orders_retail数据
    t1 = time.time()
    fdc_simu_orders_retail_pd = parse_tab_txt(path_fdc_simu_orders_retail)
    printruntime(t1, r'读取并解析 fdc_simu_orders_retail 数据')
    len_fdc_simu_orders_retail_pd = len(fdc_simu_orders_retail_pd.index)
    fdc_simu_orders_retail_pd['sale_qtty'] = fdc_simu_orders_retail_pd['sale_qtty'].astype(float)
    print r'[ Data Anlysis ] fdc_simu_orders_retail Rows is : {0}'.format(len_fdc_simu_orders_retail_pd)

    # 读取并解析simu_orders_retail_pd数据
    t1 = time.time()
    simu_orders_retail_pd = parse_tab_txt(path_simu_orders_retail)
    printruntime(t1, r'读取并解析 simu_orders_retail 数据')
    simu_orders_retail_pd['sale_qtty'] = simu_orders_retail_pd['sale_qtty'].astype(float)
    len_simu_orders_retail_pd = len(simu_orders_retail_pd.index)
    print r'[ Data Anlysis ] simu_orders_retail Rows is : {0}'.format(len_simu_orders_retail_pd)

    # 4获取FDC的销量和全部RDC的销量 仿真
    t1 = time.time()
    printruntime(t1, r'获取FDC的销量和全部RDC的销量 仿真')
    tmp_fdc_simu_orders_retail = fdc_simu_orders_retail_pd['sale_qtty'].groupby(
            [fdc_simu_orders_retail_pd["date_s"], fdc_simu_orders_retail_pd["fdc"], fdc_simu_orders_retail_pd["sku"]]).sum()
    tmp_fdc_simu_orders_retail = tmp_fdc_simu_orders_retail.reset_index()
    tmp_simu_orders_retail_pd = simu_orders_retail_pd['sale_qtty'].groupby(
            [simu_orders_retail_pd["date_s"], simu_orders_retail_pd["fdc"], simu_orders_retail_pd["sku"]]).sum()
    tmp_simu_orders_retail_pd = tmp_simu_orders_retail_pd.reset_index()
    # 获取实际销量数据.......

    t1 = time.time()
    start_date = '2016-10-02'
    end_date = '2016-10-31'
    date_range = datelist(start_date, end_date)
    # 读入订单明细数据
    print '开始读取明细数据....'
    pkl_sale = []
    for p in date_range:
        print p
        pkl_sale_mid = open(sale_data_path + p + '.pkl')
        mid_allocation_sale_data = pickle.load(pkl_sale_mid)
        pkl_sale.append(mid_allocation_sale_data)
        pkl_sale_mid.close()
    allocation_sale_data = pd.concat(pkl_sale)
    allocation_sale_data.columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id', 'item_sku_id',
                                    'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'white_flag_01',
                                    'item_third_cate_cd',
                                    'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    tmp_allocation_sale_data = allocation_sale_data['sale_qtty'].groupby(
            [allocation_sale_data["date_s"], allocation_sale_data["dc_id"] , allocation_sale_data["item_sku_id"]]).sum()
    tmp_allocation_sale_data = tmp_allocation_sale_data.reset_index()

    printruntime(t1, r'获取FDC的销量和全部RDC的销量 仿真')
    ##3，获取预测数据
    print '开始读取SKU预测数据....'
    t1 = time.time()
    pkl_sku = []
    start_date = '2016-10-02'
    end_date = '2016-10-31'
    date_range = datelist(start_date, end_date)
    for p in date_range:
        print p
        pkl_sku_mid = open(sku_data_path + p + '.pkl')
        mid_allocation_sku_data = pickle.load(pkl_sku_mid)
        mid_allocation_sku_data.columns = ['sku_id', 'mean_sales', 'variance', 'ofdsales', 'inv', 'white_flag',
                                           'white_flag_01', 'date_s', 'dc_id', 'variance_ofdsales', 'std']
        mid_allocation_sku_data = mid_allocation_sku_data.loc[:, ['sku_id', 'date_s', 'dc_id', 'mean_sales', 'std']]
        mid_allocation_sku_data['sku_id'] = mid_allocation_sku_data['sku_id'].astype(str)
        mid_allocation_sku_data = pd.merge(mid_allocation_sku_data, sample_sku_list, left_on=['sku_id'], right_on=['sku'])
        print mid_allocation_sku_data
        pkl_sku.append(mid_allocation_sku_data)
        pkl_sku_mid.close()
    allocation_sku_data_sample = pd.concat(pkl_sku)
    printruntime(t1, r'获取预测数据')
    # print allocation_sale_data[allocation_sale_data['sku_id']==1468837]
    # 合并抽取相关数据
    print '合并前的数据情况', sample_sku_list.shape, fdc_inv_pd.shape
    tmp_01 = pd.merge(sample_sku_list, fdc_inv_pd, on=['sku'])
    tmp_02 = pd.merge(tmp_01, tmp_fdc_simu_orders_retail, on=['date_s', 'sku', 'fdc'], how='left')
    tmp_02 = tmp_02.loc[:, ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po", "sale_qtty"]]
    tmp_02.columns = ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po", "sale_qtty_fdc"]
    tmp_03 = pd.merge(tmp_02, tmp_simu_orders_retail_pd, on=['date_s', 'sku', 'fdc'], how='left')
    tmp_03 = tmp_03.loc[:,
                    ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po", "sale_qtty_fdc", "sale_qtty"]]
    tmp_03.columns = ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po", "sale_qtty_fdc",
                    "sale_qtty_rdc_simu"]
    allocation_sale_data = tmp_allocation_sale_data.astype(str)
    tmp_04 = pd.merge(tmp_03, allocation_sale_data, left_on=['date_s', 'sku', 'fdc'],
                    right_on=['date_s', 'item_sku_id', 'dc_id'], how='left')
    print tmp_04[tmp_04['sku'] == '3107131']
    tmp_04 = tmp_04.loc[:, ["date_s", "fdc", "sku", "inv", "allocation", "arrive_quantity", "open_po", "sale_qtty_fdc",
                    "sale_qtty_rdc_simu", "sale_qtty"]]
    # 库存数据为当天的末尾数据，改为头部数据
    tmp_04 = tmp_04.fillna(0)
    tmp_04['inv'] = tmp_04['inv'] + tmp_04['sale_qtty_fdc']
    print tmp_04[tmp_04['sku'] == '3107131']
    # 转换数据
    allocation_sku_data_sample['mean_sales'] = allocation_sku_data_sample['mean_sales'].apply(to_num_value)
    allocation_sku_data_sample['std'] = allocation_sku_data_sample['std'].apply(to_num_value)
    tmp_05 = pd.merge(tmp_04, allocation_sku_data_sample, left_on=['date_s', 'sku', 'fdc'],
                    right_on=['date_s', 'sku_id', 'dc_id'], how='left')
    print '输出相关数据'
    tmp_05.to_csv(save_path + os.sep + 'table_sample_sku.csv', index=False)
    print '完成！'

