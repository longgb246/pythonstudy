#-*- coding:utf-8 -*-
import os
import sys
import pickle
import pandas as pd
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import configServer as conf
path = conf.data_path
total_path = path + os.sep + 'data_total3'


def mkdirpath(total_path):
    if os.path.exists(total_path) == False:
        os.mkdir(total_path)


def transfer_fdc(path):
    read_data_path = path + os.sep + 'fdc_datasets' + os.sep + 'dev_allocation_fdc_data'
    read_data_file = read_data_path + os.sep + '000000_0'
    read_data = pd.read_table(read_data_file, sep='\001', header=None)
    read_data.columns = ['org_from', 'org_to', 'actiontime_max', 'alt_max', 'alt_cnt']
    with open(total_path + os.sep + 'fdc_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_sku(path):
    read_data_path = path + os.sep + 'sku_datasets' + os.sep + 'dev_allocation_sku_data'
    data_columns = ['sku_id', 'forecast_begin_date', 'forecast_days', 'forecast_daily_override_sales',
		'forecast_weekly_override_sales', 'forecast_weekly_std', 'forecast_daily_std', 'variance',
		'ofdsales' , 'inv', 'arrive_quantity', 'open_po', 'white_flag', 'date_s', 'dc_id']
    # 1、遍历日期 date
    flag = 0
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
                            if flag == 0:
                                read_data = read_data_tmp
                                flag += 1
                            else:
                                read_data = pd.concat([read_data, read_data_tmp])
    with open(total_path + os.sep + 'sku_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_order(path):
    read_data_path = path + os.sep + 'order_datasets' + os.sep + 'dev_allocation_order_data'
    data_columns = ['arrive_time','item_sku_id','arrive_quantity','rdc_id']
    # 1、遍历 rdc_id
    flag = 0
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
                    if flag == 0:
                        read_data = read_data_tmp
                        flag += 1
                    else:
                        read_data = pd.concat([read_data, read_data_tmp])
    with open(total_path + os.sep + 'order_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)
    return read_data


def transfer_sale(path):
    read_data_path = path + os.sep + 'sale_datasets' + os.sep + 'dev_allocation_sale_data'
    data_columns = ['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id','item_sku_id',
                    'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag', 'item_third_cate_cd',
                    'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
    # 1、遍历日期 date
    flag = 0
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
                            if flag == 0:
                                read_data = read_data_tmp
                                flag += 1
                            else:
                                read_data = pd.concat([read_data, read_data_tmp])
    with open(total_path + os.sep + 'sale_data.pkl', 'wb') as f:
        pickle.dump(read_data, f)


if __name__ == '__main__':
    mkdirpath(total_path)
    print "transfer_fdc..."
    transfer_fdc(path)
    print "transfer_sku..."
    transfer_sku(path)
    print "transfer_order..."
    transfer_order(path)
    print "transfer_sale..."
    transfer_sale(path)
    # f = open(r'/home/cmo_ipc/Allocation_shell/datasets/data_total/fdc_data.pkl','rb')
    # data = pickle.load(f)


