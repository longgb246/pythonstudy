#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import time
import cPickle as pickle
import re


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


# 读取文件路径
path_analysis = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3'
path_fdc_inv = path_analysis + os.sep + 'fdc_inv.pkl'
path_fdc_simu_orders_retail = path_analysis + os.sep + 'fdc_simu_orders_retail.txt'
sale_data_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sale/'
sku_data_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_fdcinv'

read_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/longgb/statistics_data'
save_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/longgb/statistics_data'


if __name__ == '__main__':
    # ===========================================================================
    # =                             （1）读取并解析数据                          =
    # ===========================================================================
    # 抽取sku，pre4的样本
    sample_sku_list = pd.read_csv(read_path + os.sep + '628_per4.csv')
    sample_sku_list = sample_sku_list['sku_id']
    sample_sku_list.columns = ['sku_id']
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

    # 2，获取销量数据，包括仿真FDC销量，仿真总销量，实际总销量
    # 读取并解析fdc_simu_orders_retail数据
    t1 = time.time()
    fdc_simu_orders_retail_pd = parse_tab_txt(path_fdc_simu_orders_retail)
    printruntime(t1, r'读取并解析 fdc_simu_orders_retail 数据')
    len_fdc_simu_orders_retail_pd = len(fdc_simu_orders_retail_pd.index)
    fdc_simu_orders_retail_pd['sale_qtty'] = fdc_simu_orders_retail_pd['sale_qtty'].astype(float)
    fdc_simu_orders_retail_pd = fdc_simu_orders_retail_pd['sale_qtty'].groupby(
        [fdc_simu_orders_retail_pd["date_s"], fdc_simu_orders_retail_pd["fdc"], fdc_simu_orders_retail_pd["sku"]])\
        .sum().reset_index()
    print r'[ Data Anlysis ] fdc_simu_orders_retail Rows is : {0}'.format(len_fdc_simu_orders_retail_pd)

    # 获取实际销量数据.......
    t1 = time.time()
    start_date = '2016-10-02'
    end_date = '2016-10-31'
    date_range = getdaterange(start_date, end_date)
    # 读入订单明细数据
    # print '开始读取明细数据....'
    pkl_sale = []
    fdc_inv_in = []
    for p in date_range:
        t2 = time.time()
        print p
        pkl_sale_mid = open(sale_data_path + p + '.pkl')
        mid_allocation_sale_data = pickle.load(pkl_sale_mid)
        pkl_sale.append(mid_allocation_sale_data)
        pkl_sale_mid.close()
        # fdc_inv
        allocation_fdc_initialization = pickle.load(open(sale_data_path + os.sep + '{0}.pkl'.format(p)))
        allocation_fdc_initialization.columns = ['sku', 'open_po_fdc_actual', 'inv_actual', 'date_s', 'fdc']
        allocation_fdc_initialization = allocation_fdc_initialization.merge(sample_sku_list, on=['sku'], how='inner')
        fdc_inv_in.append(allocation_fdc_initialization)
        printruntime(t2, '读取一天的时间')
    fdc_inv_actual = pd.concat(fdc_inv_in)
    allocation_sale_data = pd.concat(pkl_sale)
    allocation_sale_data.columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id', 'item_sku_id',
                                    'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'white_flag_01',
                                    'item_third_cate_cd', 'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    tmp_allocation_sale_data = allocation_sale_data['sale_qtty'].groupby(
        [allocation_sale_data["date_s"], allocation_sale_data["dc_id"], allocation_sale_data["item_sku_id"]]).sum()
    tmp_allocation_sale_data = tmp_allocation_sale_data.reset_index()
    tmp_allocation_sale_data.columns = ['date_s', 'fdc', 'sku', 'sale_qtty_actual']
    printruntime(t1, r'获取FDC的销量和全部RDC的销量 仿真')
    tmp_allocation_sale_data = tmp_allocation_sale_data.merge(sample_sku_list, on=['sku'], how='inner')

    fdc_inv_compare = fdc_inv.merge(fdc_inv_actual, on=['sku_id', 'dc_id','date_s'])
    sale_compare = fdc_simu_orders_retail_pd.merge(tmp_allocation_sale_data, on=['sku_id', 'dc_id','date_s'])

    fdc_inv_compare.to_csv(save_path + os.sep + 'fdc_inv_compare.csv', index=False)
    sale_compare.to_csv(save_path + os.sep + 'sale_compare.csv', index=False)



    # 抽数据问题
    # 这个地方的问题?
    path2 = r'D:\Lgb\data_rz\628_per4.csv'
    data_tmp = pd.read_csv(path2)
    data_tmp_upper = data_tmp[data_tmp['delta'] > 0]
    data_tmp_lower = data_tmp[data_tmp['delta'] < 0]
    # data_tmp[data_tmp['sku_id'] == 2967897]
    np.random.choice(data_tmp_upper['sku_id'],20)
    # [ 1280273,  928128, 2736969, 3303104,  818098,  189982, 2967897, 2956045, 2857483, 3093094, 2131674,  526835, 1150552, 1043580, 2141606, 2247316,  923635, 2655623, 1084057, 3343745]
    np.random.choice(data_tmp_lower['sku_id'],20)
    # [ 852297,  884573,  246463, 1947864,  689229, 1039901, 2011989, 2202891, 3029042, 1450592,  688044,  688044,  312601,  385631, 3508620, 1085712,  766246, 3749844,  385665, 2069538]
    data_tmp_upper.to_csv(r'D:\Lgb\data_rz\628_per4_upper.csv', index=False)
    data_tmp_lower.to_csv(r'D:\Lgb\data_rz\628_per4_lower.csv', index=False)

