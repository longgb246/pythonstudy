#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import cPickle as pickle


def getdaterange(start_date,end_date):
    '''
    生成日期，格式'yyyy-mm-dd'
    '''
    # date_format = '{0}-{1}-{2}'
    # date_range = map(lambda x: date_format.format(str(x)[:4], str(x)[5:7], str(x)[8:10]),pd.date_range(start_date, end_date).values)
    date_range = map(lambda x: str(x)[:10],pd.date_range(start_date, end_date).values)
    return date_range


# 设置路径
# sale_data_path = r'/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total3/total_sale'
sale_data_path = r'D:\Lgb\rz_sz'
save_path = r'D:\Lgb\rz_sz'

# 设置时间
start_date = '2016-10-02'
end_date = '2016-10-04'
date_range = getdaterange(start_date, end_date)


if __name__ == '__main__':
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

    # sql中第一层
    allocation_sale_data['sale_ord_white_flag_map'] = map(lambda x: 1 if (x == 1) or (x == '1') else 0,allocation_sale_data['sale_ord_white_flag'].values)
    allocation_sale_data['sale_ord_type_map'] = map(lambda x: 1 if (x == 'rdc') else 0,allocation_sale_data['sale_ord_type'].values)
    allocation_sale_data_tmp = allocation_sale_data.loc[:,['date_s', 'dc_id', 'parent_sale_ord_id', 'sale_ord_id', 'sale_qtty', 'sale_ord_white_flag_map', 'sale_ord_type_map']]
    allocation_sale_data_tmp2 = allocation_sale_data_tmp.groupby(['date_s', 'dc_id', 'parent_sale_ord_id'])\
        .agg({'sale_ord_id': 'count', 'sale_qtty': np.sum, 'sale_ord_white_flag_map': np.sum, 'sale_ord_type_map': np.sum})\
        .reset_index()

    # sql中第二层
    if_all_white_mask = allocation_sale_data_tmp2['sale_ord_id'] == allocation_sale_data_tmp2['sale_ord_white_flag_map']
    delivery_type_mask = allocation_sale_data_tmp2['sale_ord_id'] == allocation_sale_data_tmp2['sale_ord_type_map']
    allocation_sale_data_tmp2['if_all_white'] = map(lambda x: 1 if x else 0, if_all_white_mask.values)
    allocation_sale_data_tmp2['delivery_type'] = map(lambda x: 'rdc' if x else 'other', delivery_type_mask.values)
    allocation_sale_data_tmp2.to_csv(save_path + os.sep + 'dev_parent_order.csv', index=False)

    test = allocation_sale_data_tmp2.loc[(allocation_sale_data_tmp2['date_s'] == '2016-10-02') & (allocation_sale_data_tmp2['dc_id'] == '628'), :]
    Counter(test['delivery_type'] == 'other')

