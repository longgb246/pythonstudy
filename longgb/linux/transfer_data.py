#-*- coding:utf-8 -*-
import os
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import configServer as conf
path = conf.data_path
total_path = path + os.sep + 'data_total3'
total_path_sku = total_path + os.sep + 'total_sku'
NaN = 'NaN'

def data_get(x):
    '''
    数据转化函数 1
    '''
    if (x == '\N') or (x == 'NaN'):
        x = np.nan
    else:
        x = eval(x)
        if x == 'NaN':
            return np.nan
        else:
            x = str(x[:7])
    return x


def data_trans2(x):
    '''
    数据转化函数 2
    '''
    x = str(x)
    if (',' in x) and ('[' in x) and (']' in x):
        data = x.split(',')
        dataset = [data[0][1:]] + data[1:7]
        return dataset
    else:
        return np.nan


def mkdirpath(total_path):
    '''
    创建总文件夹
    '''
    if os.path.exists(total_path) == False:
        os.mkdir(total_path)


def transfer_fdc(path):
    '''
    转化 fdc 的数据
    '''
    read_data_path = path + os.sep + 'fdc_datasets' + os.sep + 'dev_allocation_fdc_data'
    read_data_file = read_data_path + os.sep + '000000_0'
    read_data = pd.read_table(read_data_file, sep='\001', header=None)
    read_data.columns = ['org_from', 'org_to', 'actiontime_max', 'alt_max', 'alt_cnt']
    with open(total_path + os.sep + 'fdc_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_sku(path):
    '''
    转化 sku 的数据
    '''
    read_data_path = path + os.sep + 'sku_datasets' + os.sep + 'dev_allocation_sku_data'
    data_columns = ['sku_id', 'forecast_begin_date', 'forecast_days', 'forecast_daily_override_sales',
                    'forecast_weekly_override_sales', 'forecast_weekly_std', 'forecast_daily_std', 'variance',
                    'ofdsales' , 'inv', 'arrive_quantity', 'open_po', 'white_flag', 'date_s', 'dc_id']
    select_columns = ['sku_id','forecast_daily_override_sales','variance','ofdsales','white_flag','date_s','dc_id']
    # 1、遍历日期 date
    flag = 0
    sku_data_list=[]
    for each_date in os.listdir(read_data_path):
        print each_date
        file_name_date = each_date.split('=')
        if os.path.isdir(read_data_path + os.sep + each_date):
            read_data_path_date = read_data_path + os.sep + each_date
            # 2、遍历 dc_id
            for each_dcid in os.listdir(read_data_path_date):
                file_name_dcid = each_dcid.split('=')
                if (os.path.isdir(read_data_path_date + os.sep + each_dcid)) and (len(file_name_dcid[1]) < 5):
                    read_data_path_dcid = read_data_path_date + os.sep + each_dcid
                    # 3、遍历文件 files
                    for each_file in os.listdir(read_data_path_dcid):
                        data_path_this = read_data_path_dcid + os.sep + each_file
                        if os.path.isfile(data_path_this):
                            read_data_tmp = pd.read_table(data_path_this, sep='\001', header=None)
                            read_data_tmp.columns = data_columns[:-2]
                            read_data_tmp[data_columns[-2]] = file_name_date[1]
                            read_data_tmp[data_columns[-1]] = file_name_dcid[1]
                            read_data_tmp = read_data_tmp.loc[:,select_columns]
                            # read_data_tmp_select.loc[read_data_tmp_select["variance"].isnull(), ["variance"]] = '\N'
                            # read_data_tmp_select.loc[read_data_tmp_select["ofdsales"].isnull(), ["ofdsales"]] = '\N'
                            # read_data_tmp_select['variance'] = map(data_get,read_data_tmp_select["variance"].values)
                            # read_data_tmp_select['ofdsales'] = map(data_get,read_data_tmp_select["ofdsales"].values)
                            read_data_tmp['variance'] = map(data_trans2, read_data_tmp["variance"].values)
                            read_data_tmp['ofdsales'] = map(data_trans2, read_data_tmp["ofdsales"].values)
                            sku_data_list.append(read_data_tmp)
                            # if flag == 0:
                            #     read_data = read_data_tmp
                            #     flag += 1
                            # else:
    read_data = pd.concat(sku_data_list)
    with open(total_path + os.sep + 'sku_data_select_2.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_order(path):
    '''
    转化 order 的数据
    '''
    read_data_path = path + os.sep + 'order_datasets' + os.sep + 'dev_allocation_order_data'
    data_columns = ['arrive_time','item_sku_id','arrive_quantity','rdc_id']
    # 1、遍历 rdc_id
    flag = 0
    order_data_list=[]
    for each_rdcid in os.listdir(read_data_path):
        print each_rdcid
        file_name_rdcid = each_rdcid.split('=')
        if os.path.isdir(read_data_path + os.sep + each_rdcid):
            read_data_path_rdcid = read_data_path + os.sep + each_rdcid
            # 2、遍历文件 files
            for each_file in os.listdir(read_data_path_rdcid):
                data_path_this = read_data_path_rdcid + os.sep + each_file
                if os.path.isfile(data_path_this):
                    read_data_tmp = pd.read_table(data_path_this, sep='\001', header=None)
                    read_data_tmp.columns = data_columns[:-1]
                    read_data_tmp[data_columns[-2]] = file_name_rdcid[1]
                    order_data_list.append(read_data_tmp)
                    # if flag == 0:
                    #     read_data = read_data_tmp
                    #     flag += 1
                    # else:
    read_data = pd.concat(order_data_list)
    with open(total_path + os.sep + 'order_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_sale(path):
    '''
    转化 sale 的数据
    '''
    read_data_path = path + os.sep + 'sale_datasets' + os.sep + 'dev_allocation_sale_data'
    data_columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id','item_sku_id',
                    'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'item_third_cate_cd',
                    'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    # 1、遍历日期 date
    flag = 0
    sale_data_list=[]
    for each_date in os.listdir(read_data_path):
        print each_date
        file_name_date = each_date.split('=')
        if os.path.isdir(read_data_path + os.sep + each_date):
            read_data_path_date = read_data_path + os.sep + each_date
            # 2、遍历 dc_id
            for each_dcid in os.listdir(read_data_path_date):
                file_name_dcid = each_dcid.split('=')
                if (os.path.isdir(read_data_path_date + os.sep + each_dcid)) and (len(file_name_dcid[1]) < 5):
                    read_data_path_dcid = read_data_path_date + os.sep + each_dcid
                    # 3、遍历文件 files
                    for each_file in os.listdir(read_data_path_dcid):
                        data_path_this = read_data_path_dcid + os.sep + each_file
                        if os.path.isfile(data_path_this):
                            read_data_tmp = pd.read_table(data_path_this, sep='\001', header=None)
                            read_data_tmp.columns = data_columns[:-2]
                            read_data_tmp[data_columns[-2]] = file_name_date[1]
                            read_data_tmp[data_columns[-1]] = file_name_dcid[1]
                            sale_data_list.append(read_data_tmp)
                            # if flag == 0:
                            #     read_data = read_data_tmp
                            #     flag += 1
                            # else:
    read_data = pd.concat(sale_data_list)
    with open(total_path + os.sep + 'sale_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)


def test():
    '''
    仅用于测试数据的异常
    '''
    read_data_path_dcid = r'/home/cmo_ipc/Allocation_shell/datasets/sku_datasets/dev_allocation_sku_data/date_s=2016-08-07/dc_id=630'
    data_columns = ['sku_id', 'forecast_begin_date', 'forecast_days', 'forecast_daily_override_sales',
                    'forecast_weekly_override_sales', 'forecast_weekly_std', 'forecast_daily_std', 'variance',
                    'ofdsales' , 'inv', 'arrive_quantity', 'open_po', 'white_flag', 'date_s', 'dc_id']
    select_columns = ['sku_id','forecast_daily_override_sales','variance','ofdsales','white_flag','date_s','dc_id']
    file_name_date = '2016-08-04'
    file_name_dcid = '630'
    sku_data_list = []
    for each_file in os.listdir(read_data_path_dcid):
        print each_file
        data_path_this = read_data_path_dcid + os.sep + each_file
        if os.path.isfile(data_path_this):
            read_data_tmp = pd.read_table(data_path_this, sep='\001', header=None)
            read_data_tmp.columns = data_columns[:-2]
            read_data_tmp[data_columns[-2]] = file_name_date
            read_data_tmp[data_columns[-1]] = file_name_dcid
            read_data_tmp_select = read_data_tmp.loc[:, select_columns]
            # read_data_tmp_select.loc[read_data_tmp_select["variance"].isnull(), ["variance"]] = '\N'
            # read_data_tmp_select.loc[read_data_tmp_select["ofdsales"].isnull(), ["ofdsales"]] = '\N'
            # read_data_tmp_select['variance'] = map(data_get, read_data_tmp_select["variance"].values)
            # read_data_tmp_select['ofdsales'] = map(data_get, read_data_tmp_select["ofdsales"].values)
            read_data_tmp_select['variance'] = map(data_trans2, read_data_tmp_select["variance"].values)
            read_data_tmp_select['ofdsales'] = map(data_trans2, read_data_tmp_select["ofdsales"].values)
            sku_data_list.append(read_data_tmp_select)


def mkdirpath_sku(total_path_sku):
    '''
    创建 sku 总文件夹
    '''
    if os.path.exists(total_path_sku) == False:
        os.mkdir(total_path_sku)


def transfer_sku_byday(path):
    '''
    转化 sku 的数据
    '''
    read_data_path = path + os.sep + 'sku_datasets' + os.sep + 'dev_allocation_sku_data'
    data_columns = ['sku_id', 'forecast_begin_date', 'forecast_days', 'forecast_daily_override_sales',
                    'forecast_weekly_override_sales', 'forecast_weekly_std', 'forecast_daily_std', 'variance',
                    'ofdsales' , 'inv', 'arrive_quantity', 'open_po', 'white_flag', 'date_s', 'dc_id']
    select_columns = ['sku_id','forecast_daily_override_sales','variance','ofdsales','white_flag','date_s','dc_id']
    # 1、遍历日期 date
    flag = 0
    # sku_data_list=[]
    for each_date in os.listdir(read_data_path):
        print each_date
        file_name_date = each_date.split('=')
        sku_data_list = []
        if os.path.isdir(read_data_path + os.sep + each_date):
            read_data_path_date = read_data_path + os.sep + each_date
            # 2、遍历 dc_id
            for each_dcid in os.listdir(read_data_path_date):
                file_name_dcid = each_dcid.split('=')
                if (os.path.isdir(read_data_path_date + os.sep + each_dcid)) and (len(file_name_dcid[1]) < 5):
                    read_data_path_dcid = read_data_path_date + os.sep + each_dcid
                    # 3、遍历文件 files
                    for each_file in os.listdir(read_data_path_dcid):
                        data_path_this = read_data_path_dcid + os.sep + each_file
                        if os.path.isfile(data_path_this):
                            read_data_tmp = pd.read_table(data_path_this, sep='\001', header=None)
                            read_data_tmp.columns = data_columns[:-2]
                            read_data_tmp[data_columns[-2]] = file_name_date[1]
                            read_data_tmp[data_columns[-1]] = file_name_dcid[1]
                            read_data_tmp = read_data_tmp.loc[:,select_columns]
                            read_data_tmp['variance'] = map(data_trans2, read_data_tmp["variance"].values)
                            read_data_tmp['ofdsales'] = map(data_trans2, read_data_tmp["ofdsales"].values)
                            sku_data_list.append(read_data_tmp)
        read_data = pd.concat(sku_data_list)
        with open(total_path_sku + os.sep + '{0}.pkl'.format(file_name_date[1]), 'wb') as f:
            pickle.dump(read_data, f)
    return read_data


if __name__ == '__main__':
    mkdirpath(total_path)
    mkdirpath_sku(total_path_sku)
    # print "transfer_fdc..."
    #transfer_fdc(path)
    print "transfer_sku..."
    # test()
    transfer_sku_byday(path)
    # print "transfer_order..."
    # transfer_order(path)
    # print "transfer_sale..."
    # transfer_sale(path)
    # f = open(r'/home/cmo_ipc/Allocation_shell/datasets/data_total/fdc_data.pkl','rb')
    # data = pickle.load(f)


# data_trans2
# # test
# path2 = r'D:\Lgb\rz_sz\000004_0'
# data2 = pd.read_table(path, sep='\001', header=None)
# data2.columns = data_columns[:-2]
# data2["variance"] == '\N'
# data2["variance"][0]
# type(data2[data2["variance"].isnull()]["variance"].iloc[0])
# from collections import Counter
# Counter(data2["variance"].isnull())
# import numpy as np
# data2[data2["variance"].isnull()]["variance"]
# data2.loc[data2["variance"].isnull(), ["variance"]] = '\N'
# Counter(data2["variance"]=='NaN')
# data2["variance"] = map(data_trans2,data2["variance"].values)
# data2["variance"] = map(data_get,data2["variance"].values)
# data2["ofdsales"] = map(data_get,data2["ofdsales"].values)
# data2["variance"][0]
# date_s=2016-08-04


# path2 = r'D:\Lgb\rz_sz\fdc_data.pkl'
# pkl_file = open(path2, 'rb')
# data1 = pickle.load(pkl_file)
# pkl_file.close()
