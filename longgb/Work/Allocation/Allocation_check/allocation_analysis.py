#-*- coding:utf-8 -*-
import os
import cPickle as pickle
import numpy as np
import pandas as pd
import time,datetime
import re


def datelist(start, end):
    '''
    生成日期，格式'yyyy-mm-dd'
    '''
    start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(end,'%Y-%m-%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


def change_type(data, type_list):
    for each in type_list:
        data[each] = map(str, data[each].values)
    return data


def printruntime(t1, name):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d >0:
        print '[   Run Time   ] ({3}) is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print '[   Run Time   ] ({2}) is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


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
                        tmp_data.append(each[tmp_start+3].strip('\n'))          # sale_qtty
                    else:
                        tmp_split = re.findall('(.*)(630|628|658)(.*)', each[tmp_start+3])[0]
                        tmp_data.append(tmp_split[0].strip('\n'))               # sale_qtty
                    data_pd.append(tmp_data)
        # print len(data_pd)
        # print data_pd[0]
        data_pd = pd.DataFrame(data_pd, columns=data_name)
    return data_pd


# ===========================================================================
# =                                 参数设置                                 =
# ===========================================================================
# 读取文件路径
path_analysis = r'/data0/cmo_ipc/inv_opt/Allocation_shell/longgb/allocation_actual_20170106_02use/result'
# /data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3
path_fdc_inv = path_analysis + os.sep + 'fdc_inv.pkl'
path_fdc_simu_orders_retail = path_analysis + os.sep + 'fdc_simu_orders_retail.txt'
path_simu_orders_retail = path_analysis + os.sep + 'simu_orders_retail.txt'
fdc_initialization_inv_path = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_fdcinv/2016-10-02.pkl'
dev_parent_order = '/data0/cmo_ipc/inv_opt/Allocation_shell/longgb/allocation_actual_20170106_02use/result/dev_parent_order/000000_0'
# 原始订单数据
sale_data_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sale/'
# 存储文件路径
save_path = r'/data0/cmo_ipc/inv_opt/Allocation_shell/longgb/allocation_actual_20170106_02use/anlysis'
if os.path.exists(save_path) == False:
    os.mkdir(save_path)


start_date = '2016-10-02'
end_date = '2016-10-02'
date_range = getdaterange(start_date, end_date)


if __name__ == '__main__':
    # ===========================================================================
    # =                             （1）读取并解析数据                          =
    # ===========================================================================
    print r'开始读取并解析数据...'
    # 1、读取并解析fdc_inv数据
    t1 = time.time()
    fdc_inv = getpkl(path_fdc_inv)
    printruntime(t1, r'读取 fdc_inv 数据')
    t1 = time.time()
    fdc_inv_pd = parse_fdc_inv(fdc_inv)
    printruntime(t1, r'解析 fdc_inv 数据')
    len_fdc_inv_pd = len(fdc_inv_pd.index)
    print r'[ Data Anlysis ] fdc_inv Rows is : {0}'.format(len_fdc_inv_pd)

    # 2、读取并解析fdc_simu_orders_retail数据
    t1 = time.time()
    fdc_simu_orders_retail_pd = parse_tab_txt(path_fdc_simu_orders_retail)
    printruntime(t1, r'读取并解析 fdc_simu_orders_retail 数据')
    len_fdc_simu_orders_retail_pd = len(fdc_simu_orders_retail_pd.index)
    fdc_simu_orders_retail_pd['sale_qtty']=fdc_simu_orders_retail_pd['sale_qtty'].astype(float)
    print r'[ Data Anlysis ] fdc_simu_orders_retail Rows is : {0}'.format(len_fdc_simu_orders_retail_pd)

    # 3、读取并解析simu_orders_retail_pd数据
    t1 = time.time()
    simu_orders_retail_pd = parse_tab_txt(path_simu_orders_retail)
    printruntime(t1, r'读取并解析 simu_orders_retail 数据')
    simu_orders_retail_pd['sale_qtty']=simu_orders_retail_pd['sale_qtty'].astype(float)
    len_simu_orders_retail_pd = len(simu_orders_retail_pd.index)
    print r'[ Data Anlysis ] simu_orders_retail Rows is : {0}'.format(len_simu_orders_retail_pd)

    # 4、生成实际数据
    # 4.1、读取 table2：  allocation_sale_data_tmp2
    t1 = time.time()
    allocation_sale_data_tmp2 = pd.read_table(dev_parent_order, sep='\001')
    allocation_sale_data_tmp2.columns = ['date_s','dc_id','parent_sale_ord_id','sku_num','total_qtty','if_all_white','delivery_type']
    allocation_sale_data_tmp2 = change_type(allocation_sale_data_tmp2, ['dc_id','parent_sale_ord_id'])
    # print allocation_sale_data_tmp2[allocation_sale_data_tmp2['parent_sale_ord_id'] == '36381777423']
    allocation_sale_data_tmp2.to_csv(save_path + os.sep + 'big_allocation_sale_data_tmp2.csv', index=False)
    printruntime(t1, r'读取并解析 dev_parent_order 数据( table2 )')

    # 4.2、读取 table1：  allocation_sale_data
    t1 = time.time()
    pkl_sale = []
    for p in date_range:
        pkl_sale_mid = open(sale_data_path + os.sep + p + '.pkl')
        mid_allocation_sale_data = pickle.load(pkl_sale_mid)
        pkl_sale.append(mid_allocation_sale_data)
        pkl_sale_mid.close()
    allocation_sale_data = pd.concat(pkl_sale)
    allocation_sale_data.columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id', 'item_sku_id',
                            'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'white_flag_01',
                            'item_third_cate_cd', 'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    allocation_sale_data = change_type(allocation_sale_data, ['item_sku_id', 'dc_id','parent_sale_ord_id'])
    allocation_sale_data.to_csv(save_path + os.sep + 'big_allocation_sale_data.csv', index=False)
    printruntime(t1, r'读取并解析 allocation_sale_data 数据( table1 )')

    # sql中第一层
    # allocation_sale_data['sale_ord_white_flag_map'] = map(lambda x: 1 if (x == 1) or (x == '1') else 0,allocation_sale_data['sale_ord_white_flag'].values)
    # allocation_sale_data['sale_ord_type_map'] = map(lambda x: 1 if (x == 'rdc') else 0,allocation_sale_data['sale_ord_type'].values)
    # allocation_sale_data_tmp = allocation_sale_data.loc[:,['date_s', 'dc_id', 'parent_sale_ord_id', 'sale_ord_id', 'sale_qtty', 'sale_ord_white_flag_map', 'sale_ord_type_map']]
    # allocation_sale_data_tmp2 = allocation_sale_data_tmp.groupby(['date_s', 'dc_id', 'parent_sale_ord_id'])\
    #     .agg({'sale_ord_id': 'count', 'sale_qtty': np.sum, 'sale_ord_white_flag_map': np.sum, 'sale_ord_type_map': np.sum})\
    #     .reset_index()
    # # sql中第二层
    # if_all_white_mask = allocation_sale_data_tmp2['sale_ord_id'] == allocation_sale_data_tmp2['sale_ord_white_flag_map']
    # delivery_type_mask = allocation_sale_data_tmp2['sale_ord_id'] == allocation_sale_data_tmp2['sale_ord_type_map']
    # allocation_sale_data_tmp2['if_all_white'] = map(lambda x: 1 if x else 0, if_all_white_mask.values)
    # allocation_sale_data_tmp2['delivery_type'] = map(lambda x: 'rdc' if x else 'other', delivery_type_mask.values)

    # 4.3、读取 table3：  fdc_simu_orders_retail_pd_merge
    t1 = time.time()
    mask = fdc_simu_orders_retail_pd['sale_qtty']>0
    tmp_fdc_simu_orders_retail = fdc_simu_orders_retail_pd[mask]
    fdc_simu_orders_retail_pd_merge = tmp_fdc_simu_orders_retail.copy()
    fdc_simu_orders_retail_pd_merge.columns = ['dc_id','date_s','sale_ord_tm','parent_sale_ord_id','sku','sale_qtty']
    fdc_simu_orders_retail_pd_merge = change_type(fdc_simu_orders_retail_pd_merge, ['sku', 'dc_id','parent_sale_ord_id'])
    fdc_simu_orders_retail_pd_merge = fdc_simu_orders_retail_pd_merge.loc[fdc_simu_orders_retail_pd_merge['date_s'] == '2016-10-02', :]
    fdc_simu_orders_retail_pd_merge.to_csv(save_path + os.sep + 'big_fdc_simu_orders_retail_pd_merge.csv', index=False)
    printruntime(t1, r'读取并解析 fdc_simu_orders_retail_pd 数据( table3 )')

    # 合并 table1 - table2 - table3
    allocation_sale_data_select = allocation_sale_data.merge(allocation_sale_data_tmp2, on=['date_s', 'dc_id', 'parent_sale_ord_id'], how='inner')
    # print '\n','before...............'
    # print allocation_sale_data_select[allocation_sale_data_select['parent_sale_ord_id'] == '36381777423']
    # 仅仅取出 ['delivery_type'] == 'other'
    # allocation_sale_data_select = allocation_sale_data_select[allocation_sale_data_select['delivery_type'] == 'other']
    # print '\n','after...............'
    # print allocation_sale_data_select[allocation_sale_data_select['parent_sale_ord_id'] == '36381777423']
    allocation_sale_data_select = allocation_sale_data_select.rename(columns={'item_sku_id': 'sku'})
    # allocation_sale_data_select.to_csv(save_path + os.sep + 'allocation_sale_data_select.csv', index=False)
    allocation_sale_data_select = allocation_sale_data_select.merge(fdc_simu_orders_retail_pd_merge, on=['date_s', 'dc_id', 'sku', 'parent_sale_ord_id'], how='outer')
    allocation_sale_data_select.to_csv(save_path + os.sep + 'allocation_sim_actual_select.csv', index=False)

    # 4.4、读取 table4：  fdc_inv_pd_copy
    t1 = time.time()
    fdc_inv_pd_copy = fdc_inv_pd.copy()
    mask = fdc_inv_pd_copy['inv']>0
    fdc_inv_pd_copy = fdc_inv_pd_copy[mask]
    mask = fdc_inv_pd_copy['date_s'] == '2016-10-02'
    fdc_inv_pd_copy = fdc_inv_pd_copy[mask]
    fdc_inv_pd_copy = fdc_inv_pd_copy.rename(columns={'fdc': 'dc_id'})
    fdc_inv_pd_copy = change_type(fdc_inv_pd_copy, ['sku', 'dc_id'])
    fdc_inv_pd_copy.to_csv(save_path + os.sep + 'big_allocation_sim_actual_select.csv', index=False)
    printruntime(t1, r'读取并解析 fdc_inv_pd 数据( table4 )')

    # 4.5、读取 table5：  allocation_fdc_initialization
    t1 = time.time()
    pkl_fdc_initialization = open(fdc_initialization_inv_path)
    allocation_fdc_initialization = pickle.load(pkl_fdc_initialization)
    allocation_fdc_initialization.columns = ['sku', 'open_po_fdc_actual', 'inv_actual', 'date_s', 'dc_id']
    allocation_fdc_initialization = change_type(allocation_fdc_initialization, ['sku', 'dc_id'])
    allocation_fdc_initialization.to_csv(save_path + os.sep + 'big_allocation_fdc_initialization.csv', index=False)
    printruntime(t1, r'读取并解析 allocation_fdc_initialization 数据( table5 )')

    # 合并 table4 - table5
    allocation_sale_data_select = allocation_sale_data_select.merge(fdc_inv_pd_copy, on=['date_s', 'dc_id', 'sku'], how='left')
    allocation_sale_data_select = allocation_sale_data_select.merge(allocation_fdc_initialization, on=['date_s', 'dc_id', 'sku'], how='left')
    allocation_sale_data_select.to_csv(save_path + os.sep + 'allocation_sim_actual_select_sale_inv2.csv', index=False)


    # ===========================================================================
    # =                             （2）生成周转、满足率表                       =
    # ===========================================================================
    t1 = time.time()
    # 4.1、生成ito周转表，含sku，销量的数值均为1 估计不对
    tmp_fdc_simu_orders_retail = fdc_simu_orders_retail_pd['sale_qtty'].groupby([fdc_simu_orders_retail_pd["date_s"],fdc_simu_orders_retail_pd["fdc"]
                                                                                  ,fdc_simu_orders_retail_pd["sku"]]).sum()
    tmp_fdc_simu_orders_retail = tmp_fdc_simu_orders_retail.reset_index()
    # print tmp_fdc_simu_orders_retail.head()
    mask = fdc_inv_pd['inv']>0
    fdc_inv_pd = fdc_inv_pd[mask]
    table_ito_sku = fdc_inv_pd.merge(tmp_fdc_simu_orders_retail, on=["date_s","fdc","sku"], how='left')
    table_ito_sku.fillna(0,inplace=True)
    print '管理之后的值'
    # print table_ito_sku.head()
    table_ito_sku = table_ito_sku.loc[:,["date_s","fdc","inv","sale_qtty", "sku"]]
    print '选择指定列之后的值'
    # print table_ito_sku.head()
    table_ito_sku["inv"] = map(float, table_ito_sku["inv"].values)
    table_ito_sku["sale_qtty"] = map(float, table_ito_sku["sale_qtty"].values)
    table_ito_sku['inv'] = table_ito_sku['inv'] + table_ito_sku['sale_qtty']
    print 'table_ito_sku.head.....is follow'
    # print table_ito_sku.head()

    # 4.2、生成ito周转表，不含sku,开始这个表生成的数据是错误的~~~~12.27修改
    tmp_table_ito_sku = table_ito_sku.loc[:,["date_s","fdc","inv","sale_qtty"]]
    # print table_ito_sku.head()
    table_ito = tmp_table_ito_sku.groupby(["date_s","fdc"]).sum()
    print 'the test result....'
    # print table_ito.head()
    table_ito["sale_qtty"] = map(float, table_ito["sale_qtty"].values)
    table_ito["ito_rate"] = table_ito["inv"] / (table_ito["sale_qtty"] + 0.0000001)
    table_ito = table_ito.reset_index()

    # 4.3、生成sku周转表，不包含日期
    tmp_table_ito_sku_nodate = table_ito_sku.loc[:,["fdc","inv","sale_qtty","sku"]]
    # print table_ito_sku.head()
    table_ito_sku_nodate = tmp_table_ito_sku_nodate.groupby(["sku","fdc"]).sum()
    print 'the test result....'
    # print table_ito_sku_nodate.head()
    table_ito_sku_nodate["sale_qtty"] = map(float, table_ito_sku_nodate["sale_qtty"].values)
    table_ito_sku_nodate["ito_rate"] = table_ito_sku_nodate["inv"] / (table_ito_sku_nodate["sale_qtty"] + 0.0000001)
    table_ito_sku_nodate=table_ito_sku_nodate.reset_index()

    # 5、生成fill满足率表
    mask = fdc_simu_orders_retail_pd['sale_qtty']>0
    tmp_fdc_simu_orders_retail = fdc_simu_orders_retail_pd[mask]
    tmp_table_fill_01 = fdc_simu_orders_retail_pd['sale_ord_id'].groupby([fdc_simu_orders_retail_pd["fdc"], fdc_simu_orders_retail_pd["date_s"]]).nunique().reset_index()
    tmp_table_fill_01.columns = ["fdc", "date_s", "fdc_order_num"]
    # print tmp_table_fill_01.head()
    mask = simu_orders_retail_pd['sale_qtty']>0
    simu_orders_retail_pd = simu_orders_retail_pd[mask]
    tmp_table_fill_02 = simu_orders_retail_pd['sale_ord_id'].groupby([simu_orders_retail_pd["date_s"],simu_orders_retail_pd["fdc"]]).nunique().reset_index()
    tmp_table_fill_02.columns = ["date_s", "fdc", "total_order_num"]
    table_fill = tmp_table_fill_01.merge(tmp_table_fill_02, on=["fdc", "date_s"], how='inner')
    table_fill["fill_rate"] = table_fill["fdc_order_num"] * 1.0 / table_fill["total_order_num"]
    printruntime(t1, '生成 table_ito_sku、table_ito、table_fill 表')


    # ===========================================================================
    # =                  （3）因为调拨不合理有多少订单变成了无效订单               =
    # ===========================================================================
    start_date = '2016-10-02'
    end_date = '2016-10-31'
    date_range = datelist(start_date, end_date)
    # 读入订单明细数据
    '''
    date,sale_ord_id,item_sku_id,sale_qtty,sale_ord_tm,sale_ord_type,sale_ord_white_flag
    '''
    pkl_sale = []
    for p in date_range:
        pkl_sale_mid = open(sale_data_path + p + '.pkl')
        mid_allocation_sale_data = pickle.load(pkl_sale_mid)
        pkl_sale.append(mid_allocation_sale_data)
        pkl_sale_mid.close()
    allocation_sale_data = pd.concat(pkl_sale)
    allocation_sale_data.columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id','item_sku_id',
                                  'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag','white_flag_01', 'item_third_cate_cd',
                                  'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    # 判断订单是否一致，订单编号，sku,sale_qtty完全一样的订单为合格订单，其余均为不一样
    # ['fdc','date_s','sale_ord_tm','sale_ord_id','sku','sale_qtty']---simu_orders_retail_pd
    sale_all_same_df = pd.merge(allocation_sale_data, simu_orders_retail_pd, how='left',
                              left_on=['dc_id','date_s','sale_ord_tm','parent_sale_ord_id','item_sku_id','sale_qtty'],
                              right_on=['fdc','date_s','sale_ord_tm','sale_ord_id','sku','sale_qtty'])
    sale_all_same_df = sale_all_same_df.fillna(0)
    mask = sale_all_same_df['sale_qtty']>0
    # print sale_all_same_df[mask].head()


    # ===========================================================================
    # =                             （4）保存数据表                              =
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
    table_ito_sku_nodate.to_csv(save_path + os.sep + 'table_ito_sku_nodate.csv', index=False)
    printruntime(t1, '生成 table_ito_sku、table_ito、table_fill 表')
    print '[ Data Anlysis ] Finish !'



#
# # Windows下运行
#
# # test : 36381777423 \ 36220673521
# # print allocation_sale_data_select[allocation_sale_data_select['parent_sale_ord_id'] == '36381777423']
#
# # 0、设置路径
# path = r'D:\Lgb'
# path_tmp = path + os.sep + 'big_allocation_sale_data_tmp2.csv'
# path_tmp = path + os.sep + 'allocation_sale_data_tmp2_big.csv'
# path_sale = path + os.sep + 'big_allocation_sale_data.csv'
# path_sale = path + os.sep + 'allocation_sale_data_big.csv'
# path_fdc_simu_orders_retail_pd_merge = path + os.sep + 'big_fdc_simu_orders_retail_pd_merge.csv'
# path_allocation_sim_actual_select = path + os.sep + 'big_allocation_sim_actual_select.csv'
# path_allocation_fdc_initialization = path + os.sep + 'big_allocation_fdc_initialization.csv'
# # 1、读取数据
# path_tmp = pd.read_csv(path_tmp)
# path_sale = pd.read_csv(path_sale)
# fdc_simu_orders_retail_pd_merge = pd.read_csv(path_fdc_simu_orders_retail_pd_merge)
# fdc_inv_pd_copy = pd.read_csv(path_allocation_sim_actual_select)
# allocation_fdc_initialization = pd.read_csv(path_allocation_fdc_initialization)
# # 2、格式转换
# path_sale = change_type(path_sale, ['item_sku_id', 'dc_id', 'parent_sale_ord_id'])
# path_tmp = change_type(path_tmp, ['dc_id', 'parent_sale_ord_id'])
# fdc_simu_orders_retail_pd_merge = change_type(fdc_simu_orders_retail_pd_merge, ["dc_id", 'parent_sale_ord_id', 'sku'])
# fdc_inv_pd_copy = change_type(fdc_inv_pd_copy, ["dc_id", 'sku'])
# allocation_fdc_initialization = change_type(allocation_fdc_initialization, ["dc_id", 'sku'])
# # 3、交表
# c = path_sale.merge(path_tmp, on=['date_s', 'dc_id', 'parent_sale_ord_id'], how='inner')
# c = c.rename(columns={'item_sku_id':'sku'})
# # c = c[c['delivery_type'] == 'other']
# c1 = c.merge(fdc_simu_orders_retail_pd_merge, on=['date_s', 'dc_id', 'sku', 'parent_sale_ord_id'], how='outer')
# c2 = c1.merge(fdc_inv_pd_copy, on=['date_s', 'dc_id', 'sku'], how='left')
# c3 = c2.merge(allocation_fdc_initialization, on=['date_s', 'dc_id', 'sku'], how='left')
# # 4、生成 sale_qtty_y 非全空的
# t1 = time.time()
# data_group = c3.groupby(["parent_sale_ord_id"])
# list_rest = []
# list_drop = []
# for parent_sale_ord_id, group in data_group:
#     if np.sum(group["sale_qtty_y"].notnull().values) >= 1:
#         list_rest.append(group)
#     else:
#         list_drop.append(group)
# printruntime(t1, 'group')
# # [   Run Time   ] (group) is : 0.0 min 22.0820 s
# t1 = time.time()
# result = pd.concat(list_rest)
# result_drop = pd.concat(list_drop)
# printruntime(t1, 'concat')
# [   Run Time   ] (concat) is : 0.0 min 53.2830 s
# c3[c3['parent_sale_ord_id'] == '36381777423']
# # 4、存储
# c3.to_csv(path + os.sep + 'allocation_sim_actual_select_sale_inv_final.csv', index=False)
# result.to_csv(path + os.sep + 'allocation_sale_inv_qtty_not_null.csv', index=False)
#
