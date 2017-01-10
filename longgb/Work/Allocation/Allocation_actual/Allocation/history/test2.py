#-*- coding:utf-8 -*-
import time,datetime
import pickle as pkl
import pandas as pd
from collections import  defaultdict

dapath=path_orders_retail=r'E:/Allocation_data/simu_data/orders_retail/2016-10-02.pkl'

with open(dapath) as f:
    allocation_sale_data=pkl.load(f)
    # mask=(retail_data['item_sku_id']==1468837)&(retail_data['date_s']=='2016-10-02')
    # print retail_data[mask]
    tmp_df=allocation_sale_data[['dc_id','date_s','item_sku_id','parent_sale_ord_id','sale_ord_tm','sale_qtty']]

    tmp_df=pd.DataFrame(tmp_df)
    # mask=(tmp_df['item_sku_id']==1468837)&(tmp_df['date_s']=='2016-10-02')
    mask=tmp_df['parent_sale_ord_id'].astype(str)=='36326389880'
    # mask=(tmp_df['item_sku_id']==1468837)
    tmp_df=tmp_df[mask]
    orders_retail_mid=pd.concat([tmp_df['dc_id'].astype(str)+tmp_df['date_s'].astype(str),tmp_df['sale_ord_tm'].astype(str)+
                                 tmp_df['parent_sale_ord_id'].astype(str),tmp_df[['item_sku_id','sale_qtty']]],
                                axis=1)
    orders_retail_mid.columns=['dc_date_id','id','item_sku_id','sale_qtty']
    print orders_retail_mid.shape
    # orders_retail_mid=orders_retail_mid.drop_duplicates(subset=['id','item_sku_id'])

    #orders_retail_mid=orders_retail_mid.set_index(['id','item_sku_id']).unstack(0)['sale_qtty'].to_dict()
    # orders_retail=defaultdict(lambda :defaultdict(int))
    orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    orders_retail={}
    # for f in fdc:
    #     orders_retail[f]=defaultdict(lambda :defaultdict(int))
    # orders_retail.update({row['dc_date_id']:{row['id']:{row['item_sku_id']:row['sale_qtty']}}
    #                            for index,row in orders_retail_mid.iterrows()})
    ##能直接update 是因为订单编号是唯一的 所以这个key是唯一的 不会存在覆盖的现象
    for index,row in orders_retail_mid.iterrows():
        if orders_retail.has_key(row['dc_date_id']):
            if orders_retail[row['dc_date_id']].has_key(row['id']):
                orders_retail[row['dc_date_id']][row['id']].update({row['item_sku_id']:row['sale_qtty']})
            else:
                orders_retail[row['dc_date_id']][row['id']]={row['item_sku_id']:row['sale_qtty']}
        else:
            orders_retail[row['dc_date_id']]={row['id']:{row['item_sku_id']:row['sale_qtty']}}
    for k0,v0 in orders_retail.items():
        print k0,len(v0)
        for k1,v1 in v0.items():
            # pass
            print k1,v1
