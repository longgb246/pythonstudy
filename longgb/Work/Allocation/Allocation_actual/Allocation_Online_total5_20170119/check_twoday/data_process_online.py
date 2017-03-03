# coding=utf-8
from sys import path
import os
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
import  pandas as pd
import datetime
from collections import defaultdict,OrderedDict
import cPickle as pickle
import numpy as np
# from inventory_process_online import inventory_proess
# import configServer

import time

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


def datelist(start, end):
    start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(end,'%Y-%m-%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d-%02d-%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


def gene_index(fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s)+str(fdc)+str(sku)



start_date='2016-10-02'#'2016-10-01'
end_date='2016-10-31'



sku_data_path=r'D:\Lgb\data_sz\fdc_inv\two_day_debug\sku1002\2016-10-02.pkl'
fdc_data_path=r'D:\Lgb\data_sz\fdc_inv\two_day_debug\fdc_data.pkl'
fdc_initialization_inv=r'D:\Lgb\data_sz\fdc_inv\two_day_debug\fdc_inv_intial_0102\2016-10-01.pkl'
order_data_path=r'D:\Lgb\data_sz\fdc_inv\two_day_debug\order_data.pkl'
sale_data_path=r'D:\Lgb\data_sz\fdc_inv\two_day_debug\sale_data_01_31'
save_data_path=r'D:\Lgb\data_sz\fdc_inv\two_day_debug'


t1 = time.time()
pkl_sku=open(sku_data_path)
allocation_sku_data=pickle.load(pkl_sku)
pkl_sku.close()
allocation_sku_data.columns= ['sku_id','mean_sales','variance','ofdsales','inv','white_flag','white_flag_01','date_s','dc_id','variance_ofdsales','std']
printruntime(t1, 'allocation_sku_data')       # [   Run Time   ] ( allocation_sku_data ) is : 0.0 min 56.1760 s


t1 = time.time()
pkl_fdc=open(fdc_data_path)
allocation_fdc_data=pickle.load(pkl_fdc)
pkl_fdc.close()
printruntime(t1, 'allocation_fdc_data')     # [   Run Time   ] (allocation_fdc_data) is : 0.0 min 0.2090 s


t1 = time.time()
allocation_fdc_data.columns=['org_from','org_to','actiontime_max','alt','alt_cnt']
fdc_01=allocation_fdc_data.groupby(['org_from','org_to','alt']).sum()
fdc_01=fdc_01.reset_index()
fdc_02=allocation_fdc_data['alt_cnt'].groupby([allocation_fdc_data['org_from'],allocation_fdc_data['org_to']]).sum()
fdc_02=fdc_02.reset_index()
fdc_alt=pd.merge(fdc_01,fdc_02,on=['org_from','org_to'])
fdc_alt.columns=['org_from','org_to','alt','alt_cnt','alt_all_cnt']
fdc_alt['alt_prob']=fdc_alt['alt_cnt']/fdc_alt['alt_all_cnt']
allocation_fdc_data=fdc_alt
allocation_fdc_data.columns=['org_from','dc_id','alt','alt_cnt','alt_all_cnt','alt_prob']
allocation_fdc_data['dc_id'] = map(lambda x: str(int(x)),allocation_fdc_data['dc_id'].values)
allocation_fdc_data=allocation_fdc_data[allocation_fdc_data['org_from']==316]
fdc_alt=defaultdict(list)
fdc_alt_prob=defaultdict(list)
for index,row in allocation_fdc_data.iterrows():
    if row['dc_id'] in fdc_alt:
        try:
            tmp=eval(row['alt'])
            fdc_alt[row['dc_id']].append(tmp)
        except:
            pass
    else:
        try:
            tmp=eval(row['alt'])
            fdc_alt[row['dc_id']]=[tmp]
        except:
            pass
    if row['dc_id'] in fdc_alt_prob:
        try:
            tmp=row['alt_prob']
            fdc_alt_prob[row['dc_id']].append(tmp)
        except:
            pass
    else:
        try:
            tmp=row['alt_prob']
            fdc_alt_prob[row['dc_id']]=[tmp]
        except:
            pass
printruntime(t1, 'fdc_alt_prob')            # [   Run Time   ] (fdc_alt_prob) is : 0.0 min 0.2600 s


t1 = time.time()
pkl_fdc_initialization=open(fdc_initialization_inv)
allocation_fdc_initialization=pickle.load(pkl_fdc_initialization)
pkl_sku.close()
allocation_fdc_initialization.columns=['sku_id','open_po_fdc','inv','date_s','dc_id']
allocation_fdc_initialization['date_s']=start_date
printruntime(t1, 'allocation_fdc_initialization')           # [   Run Time   ] (allocation_fdc_initialization) is : 0.0 min 13.9770 s


t1 = time.time()
pkl_order=open(order_data_path)
allocation_order_data=pickle.load(pkl_order)
pkl_order.close()
allocation_order_data.columns=['arrive_time','item_sku_id','arrive_quantity','dc_id']
printruntime(t1, 'allocation_order_data')                   # [   Run Time   ] (allocation_order_data) is : 0.0 min 5.1320 s


date_range=datelist(start_date,end_date)



#
t1 = time.time()
pkl_sale=[]
for p in [date_range[0]]:
    print p
    pkl_sale_mid=open(sale_data_path+ os.sep + p+'.pkl')
    mid_allocation_sale_data=pickle.load(pkl_sale_mid)
    pkl_sale.append(mid_allocation_sale_data)
    pkl_sale_mid.close()
allocation_sale_data=pd.concat(pkl_sale)
allocation_sale_data.columns=['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id','item_sku_id',
                              'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag','white_flag_01', 'item_third_cate_cd',
                              'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
printruntime(t1, 'allocation_sale_data')        # [   Run Time   ] (allocation_sale_data) is : 1.0 min 8.2770 s



t1 = time.time()
fdc_forecast_sales=pd.concat([allocation_sku_data['date_s'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                              +allocation_sku_data['sku_id'].astype('str'),
                              allocation_sku_data['mean_sales']],axis=1)
fdc_forecast_sales.columns=['id','forecast_value']
fdc_forecast_sales=fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()

fdc_forecast_std=pd.concat([allocation_sku_data['date_s'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                            +allocation_sku_data['sku_id'].astype('str'),
                            allocation_sku_data['std']],axis=1)
fdc_forecast_std.columns=['id','forecast_std']
fdc_forecast_std=fdc_forecast_std.set_index('id')['forecast_std'].to_dict()
printruntime(t1, 'fdc_forecast_std')            # [   Run Time   ] (fdc_forecast_std) is : 0.0 min 8.0470 s


t1 = time.time()
fdc_inv=defaultdict(lambda :defaultdict(int))
allocation_fdc_initialization['inv']=allocation_fdc_initialization['inv']+allocation_fdc_initialization['open_po_fdc']
mid_fdc_inv=pd.concat([allocation_fdc_initialization['date_s'].astype(str)+allocation_fdc_initialization['dc_id'].astype(str)
                       +allocation_fdc_initialization['sku_id'].astype(str),
                       allocation_fdc_initialization['inv']],axis=1)
mid_fdc_inv.columns=['id','inv']
mid_fdc_inv=mid_fdc_inv.drop_duplicates()
mid_fdc_inv=mid_fdc_inv.set_index('id')['inv'].to_dict()
for k,v in mid_fdc_inv.items():
    fdc_inv[k]['inv']=v
printruntime(t1, 'fdc_inv')


t1 = time.time()
white_list_dict=defaultdict(lambda :defaultdict(list))
tmp_df=allocation_sku_data[allocation_sku_data['white_flag']==1][['date_s','sku_id','dc_id']]
for k,v in tmp_df['sku_id'].groupby([tmp_df['date_s'],tmp_df['dc_id']]):
    white_list_dict[k[1]][k[0]]=list(v)#
printruntime(t1, 'white_list_dict')             # [   Run Time   ] (fdc_inv) is : 0.0 min 19.2600 s


fdc_allocation=''
fdc=['628','630','658']

t1 = time.time()
rdc_inv=defaultdict(int)
tmp_df=allocation_sku_data[allocation_sku_data['date_s']==start_date]
mid_rdc_inv=pd.concat([tmp_df['date_s'].astype(str)+'rdc'+tmp_df['sku_id'].astype(str),
                       tmp_df['inv']],axis=1)
mid_rdc_inv.columns=['id','inv']
mid_rdc_inv=mid_rdc_inv.drop_duplicates()
mid_rdc_inv=mid_rdc_inv.set_index('id')['inv'].to_dict()
rdc_inv.update(mid_rdc_inv)
printruntime(t1, 'rdc_inv')             # [   Run Time   ] (rdc_inv) is : 0.0 min 4.2670 s


t1 = time.time()
tmp_df=allocation_order_data[['arrive_time','item_sku_id','arrive_quantity']]
tmp_df.columns=['date','item_sku_id','arrive_quantity']
order_list=defaultdict(lambda :defaultdict(int))
for index,row in tmp_df.iterrows():
    if order_list.has_key(row['date']):
        if order_list[row['date']].has_key(row['item_sku_id']):
            order_list[row['date']][row['item_sku_id']]=order_list[row['date']][row['item_sku_id']]+row['arrive_quantity']
        else:
            order_list[row['date']][row['item_sku_id']]=row['arrive_quantity']
    else:
        order_list[row['date']]={row['item_sku_id']:row['arrive_quantity']}
printruntime(t1, 'order_list')          # [   Run Time   ] (order_list) is : 1.0 min 44.8280 s



t1 = time.time()
tmp_df=allocation_sale_data[['dc_id','date_s','item_sku_id','parent_sale_ord_id','sale_ord_tm','sale_qtty']]
tmp_df=pd.DataFrame(tmp_df)
orders_retail_mid=pd.concat([tmp_df['dc_id'].astype(str)+tmp_df['date_s'].astype(str),tmp_df['sale_ord_tm'].astype(str)+
                             tmp_df['parent_sale_ord_id'].astype(str),tmp_df[['item_sku_id','sale_qtty']]],
                            axis=1)
orders_retail_mid.columns=['dc_date_id','id','item_sku_id','sale_qtty']
orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
for index,row in orders_retail_mid.iterrows():
    if orders_retail.has_key(row['dc_date_id']):
        if orders_retail[row['dc_date_id']].has_key(row['id']):
            orders_retail[row['dc_date_id']][row['id']].update({row['item_sku_id']:row['sale_qtty']})
        else:
            orders_retail[row['dc_date_id']][row['id']]={row['item_sku_id']:row['sale_qtty']}
    else:
        orders_retail[row['dc_date_id']]={row['id']:{row['item_sku_id']:row['sale_qtty']}}
printruntime(t1, 'orders_retail')           # [   Run Time   ] (orders_retail) is : 0.0 min 11.2190 s


orders_retail_type=defaultdict(str)
sku_fdc_sales=defaultdict(int)
sku_rdc_sales=defaultdict(int)

all_sku_list=list(set(allocation_sku_data['sku_id'].values))
mask=allocation_order_data['arrive_time']>'2016-10-01'
tmp_allocation_order_data=allocation_order_data[mask]
all_sku_list.extend(list(set(allocation_order_data['item_sku_id'].values)))
logger = 1



del tmp_allocation_order_data
del allocation_sku_data
del allocation_fdc_data
del allocation_order_data
del allocation_fdc_initialization
del tmp_df
del allocation_sale_data

print 'Over ..... !~~'


allocation=inventory_proess(fdc_forecast_sales,fdc_forecast_std,fdc_alt,fdc_alt_prob,fdc_inv,white_list_dict,fdc_allocation,fdc,rdc_inv,
                            order_list,date_range,orders_retail,all_sku_list,logger,save_data_path)
allocation.OrdersSimulation()



pickle.dump(fdc_forecast_sales,open(save_data_path+'fdc_forecast_sales_02.pkl','w'))
pickle.dump(fdc_forecast_std,open(save_data_path+'fdc_forecast_std_02.pkl','w'))
pickle.dump(dict(allocation.fdc_inv),open(save_data_path+'fdc_inv.pkl','w'))
pickle.dump(dict(allocation.fdc_allocation),open(save_data_path+'fdc_allocation.pkl','w'))
pickle.dump(dict(allocation.rdc_inv),open(save_data_path+'rdc_inv.pkl','w'))
pickle.dump(dict(allocation.fdc_allocation),open(save_data_path+'fdc_allocation.pkl','w'))
pickle.dump(dict(allocation.rdc_inv),open(save_data_path+'rdc_inv.pkl','w'))


with open(save_data_path+'white_list_dict_02.txt','w') as white:
    for k,v in white_list_dict.items():
        for k1,v1 in v.items():
            white.write(str(k))
            white.write('\t')
            white.write(str(k1))
            white.write('\t')
            white.write(str(v1))
            white.write('\n')


with open(save_data_path+'order_list.txt','w') as ol:
    for k,v in allocation.order_list.items():
        for k1,v1 in v.items():
            ol.write(str(k))
            ol.write('\t')
            ol.write(str(k1))
            ol.write('\t')
            ol.write(str(v1))
            ol.write('\n')

with open(save_data_path+'orders_retail.txt','w') as orl:
    for k,v in allocation.orders_retail.items():
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

try:
    with open(save_data_path+'simu_orders_retail.txt','w') as orl:
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
except:
    print 'simu order  in the except'




try:
    with open(save_data_path+'allocation_retail.txt','w') as orl:
        for k,v in allocation.allocation_retail.items():
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
except:
    print 'simu order  in the except'

try:
    with open(save_data_path+'fdc_simu_orders_retail.txt','w') as orl:
        for k,v in allocation.fdc_simu_orders_retail.items():
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
    logger.info('仿真数据保存完成...仿真程序完成...')
except:
    print 'in the except'


#####计算KPI，KPI主要包括本地订单满足率，周转，SKU满足率
#本地订单满足率 (本地出库订单+订单驱动内配)/订单数量
# print 'origin orders......',allocation.orders_retail
# print 'sim orders .......',allocation.simu_orders_retail
# print 'fdc orders ......',allocation.fdc_simu_orders_retail
# print '订单满足率:.........'
# print len(allocation.fdc_simu_orders_retail)/len(allocation.simu_orders_retail)
cnt_orders_retail_type={}
for k,v in allocation.orders_retail_type.items():
    cnt_orders_retail_type.setdefault(v,[]).append(k)
for k,v in cnt_orders_retail_type.items():
    print k,'has orders number:',len(v)
#周转，考核单个SKU的周转,考察一个SKU7天的周转，7天平均库存/7天的平均销量  订单数据：{fdc_dt:{订单时间_订单id:{SKU：数量}}}
#将订单数据转换为{fdc{date：{sku,销量}}},同时需要判断订单是否有FDC出货，需要在仿真的过程中标记，便于后续获取计算
#直接标记不易标记，建立两个字典，一个记录仿真销量情况，一个记录仿真FDC销量情况
#fdc_date:{sku:数量}
sale_orders_retail_sku_cnt=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
# for f in fdc:
#     for k,v in allocation.fdc_simu_orders_retail[f].items():
#         date_sale=k[0:11]
#         for k1,v1 in v.items():
#             sale_orders_retail_sku_cnt[f][date_sale][k1]+=v1

for k,v in allocation.fdc_simu_orders_retail.items():
    k00=k[-10:]
    k01=k[:-10]
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            sale_orders_retail_sku_cnt[k01][k00][k2]+=v2
            #k,k1,k2,v2





# print allocation.fdc_inv[index]['inv'],将其拆解为{fdc:{date:{sku:inv}}}
inv_orders_retail_sku_cnt=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
for k,v in allocation.fdc_inv.items():
    k1,k2,k3=k[:11],k[11:14],k[14:] #仅针对三位数的FDC，如果采用其他的则需要考虑把FDC编码映射成三位或增加分隔符
    inv_orders_retail_sku_cnt[k1][k2][k3]=v['inv']

#遍历fdc,遍历日期，遍历sku,计算周转情况,ot_sku的数据格式：(fdc_sku_date:周转天数)
ot_sku=defaultdict(int)

for f in fdc:
    for i in range(len(date_range)):
        sub_set=date_range[i:i+7]
        for sku in all_sku_list:
            v1=0
            v2=0
            for s in sub_set:
                v1+=sale_orders_retail_sku_cnt[f][s][sku]
                v2+=inv_orders_retail_sku_cnt[f][s][sku]
            index=gene_index(f,sku,date_range[i])
            ot_sku[index]=v2/v1
print ot_sku
