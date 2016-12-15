# coding=utf-8
from sys import path
import os
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
import  pandas as pd
import datetime
from collections import defaultdict
import cPickle as pickle
import numpy as np
from com.jd.test.zhangjs.Allocation.inventory_process import inventory_proess
def datelist(start, end):
    start_date = datetime.datetime.strptime(start,'%Y%m%d')
    end_date = datetime.datetime.strptime(end,'%Y%m%d')
    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result
#指定数据相关路径
sku_data_path='E://Allocation_real_data//sku_data'
fdc_data_path='E://Allocation_real_data//fdc_data'
order_data_path='E://Allocation_real_data//order_data'
sale_data_path='E://Allocation_real_data//sale_data'
save_data_path='E://Allocation_real_data//'#数据集存储路径
#读入相关数据集
#读入SKU粒度数据集
'''date,sku_id,dc_id,forecast_begin_date,forecast_days,forecast_daily_override_sales,
forecast_weekly_override_sales,forecast_weekly_std,inv,arrive_quantity,open_po,white_flag'''
allocation_sku_data=pd.read_table(sku_data_path,sep='\001',
                                  names=['sku_id','forecast_begin_date','forecast_days','forecast_daily_override_sales',
                                         'forecast_weekly_override_sales','forecast_weekly_std','forecast_daily_std',
                                         'variance','ofdsales','inv',
                                         'arrive_quantity','open_po','white_flag'],header=None)

#读入FDC粒度数据
'''
date,dc_id,alt,alt_prob
'''
allocation_fdc_data=pd.read_table(fdc_data_path,sep='\001',
                                names=['org_from','org_to','actiontime_max','alt','alt_cnt'])
print allocation_fdc_data

#读入采购单粒度数据
'''
date,rdc,order_id,item_sku_id,arrive_quantity
'''
allocation_order_data=pd.read_table(order_data_path,sep='\001',
                                  names=['arrive_time','item_sku_id','arrive_quantity'])

#读入订单明细数据
'''
date,sale_ord_id,item_sku_id,sale_qtty,sale_ord_tm,sale_ord_type,sale_ord_white_flag
'''
allocation_sale_data=pd.read_table(sale_data_path,sep='\001',
                                 names=['org_dc_id','sale_ord_det_id','sale_ord_id','parent_sale_ord_id',
                                        'item_sku_id','sale_qtty','sale_ord_tm','sale_ord_type','sale_ord_white_flag',
                                        'item_third_cate_c','item_second_cate_cd','shelves_dt,shelves_tm'])

#定义索引转换函数
def gene_index(self,fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return date_s+fdc+sku

##将上述读入的数据集，转换为调拨仿真类需要的数据集

##标记仿真的开始和结束日期
start_date='2016/10/22'
end_date='20161201'
######增加临时分区数据##########
allocation_sku_data['date']='2016-07-01'
allocation_sku_data['dc_id']='628'
allocation_sale_data['date']='2016-07-01'
allocation_sale_data['dc_id']='628'
allocation_fdc_data['date']='2016-07-01'
allocation_fdc_data['dc_id']='628'
allocation_order_data['date']='2016-07-01'
allocation_order_data['dc_id']='628'
#预测数据相关信息{fdc_sku_date:[7 days sales]},{fdc_sku_data:[7 days cv]}
fdc_forecast_sales=pd.concat([allocation_sku_data['date'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                                 +allocation_sku_data['sku_id'].astype('str'),
                                 allocation_sku_data['forecast_daily_override_sales']],axis=1)
fdc_forecast_sales.columns=['id','forecast_value']
fdc_forecast_sales=fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()


fdc_forecast_std=pd.concat([allocation_sku_data['date'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                            +allocation_sku_data['sku_id'].astype('str'),
                            allocation_sku_data['forecast_weekly_std']],axis=1)
fdc_forecast_std.columns=['id','forecast_std']
fdc_forecast_std=fdc_forecast_std.set_index('id')['forecast_std'].to_dict()

#RDC-->FDC时长分布,{fdc:[days]}}
fdc_alt=pd.concat([allocation_fdc_data['date'].astype('str')+allocation_fdc_data['dc_id'].astype('str'),
        allocation_fdc_data['alt']],axis=1)
fdc_alt.columns=['id','alt']
fdc_alt=fdc_alt.set_index('id')['alt'].to_dict()

fdc_alt_prob=pd.concat([allocation_fdc_data['date'].astype('str')+allocation_fdc_data['dc_id'].astype('str'),
            allocation_fdc_data['alt_prob']],axis=1)
fdc_alt_prob.columns=['id','alt_prob']
fdc_alt_prob=fdc_alt_prob.set_index('id')['alt_prob'].to_dict()

#defaultdict(lamda:defaultdict(int)),只需要一个初始化库存即可
fdc_inv=defaultdict(lambda :defaultdict(int))
tmp_df=allocation_sku_data[allocation_sku_data['date']==start_date]
mid_fdc_inv=pd.concat([tmp_df['date'].astype(str)+tmp_df['dc_id'].astype(str)+tmp_df['sku_id'].astype(str),
                     tmp_df['inv']],axis=1).set_index(0)['inv'].to_dict()
for k,v in mid_fdc_inv.items():
    fdc_inv[k]['inv']=v
#白名单,不同日期的白名单不同{fdc:{date_s:[]}}
white_list_dict=defaultdict(lambda :defaultdict(list))
tmp_df=allocation_sku_data[allocation_sku_data['white_flag']==1][['date','sku_id','dc_id']]
for k,v in tmp_df['sku_id'].groupby([tmp_df['date'],tmp_df['dc_id']]):
    white_list_dict[k[0]][k[1]]=list(v)#
#调拨量字典,fdc_allocation=defaultdict(float)
fdc_allocation=''
#fdc列表：
fdc=pd.unique(allocation_fdc_data['dc_id'])
fdc=['8','682']
#RDC库存，{date_sku_rdc:库存量} defaultdict(int),只需要初始化的RDC库存
rdc_inv=defaultdict(int)
rdc_inv.update({i[0]:i[1] for i in fdc_inv.items() if 'rdc' in i[0] })
#采购单数据，采购ID，SKU，实际到达量，到达时间,将其转换为{到达时间:{SKU：到达量}}形式的字典，defaultdict(lambda :defaultdict(int))
#采购单ID在这里没有作用，更关心的是单个SKU在某一天的到达量
tmp_df=allocation_order_data[['date','item_sku_id','arrive_quantity']]
order_list=defaultdict(lambda :defaultdict(int))
order_list.update(tmp_df.set_index(['date','item_sku_id']).unstack(0)['arrive_quantity'].to_dict())

#仿真的时间窗口 时间格式如下：20161129,这个怎么使用日期区间，待定
date_range=datelist('20160101',end_date)
#订单数据：{fdc_订单时间_订单id:{SKU：数量}},当前的存储会造成的空间浪费应当剔除大小为0的SKU
tmp_df=allocation_sale_data[['dc_id','item_sku_id','sale_ord_id','sale_ord_tm','sale_qtty']]
tmp_df=tmp_df.loc[[1,2,3]]
tmp_df=pd.DataFrame(tmp_df)
orders_retail_mid=pd.concat([tmp_df['dc_id'].astype(str)+';'+tmp_df['sale_ord_tm'].astype(str)+
                             tmp_df['sale_ord_id'].astype(str),tmp_df[['item_sku_id','sale_qtty']]],
                           axis=1)
orders_retail_mid.columns=['id','item_sku_id','sale_qtty']
orders_retail_mid=orders_retail_mid.set_index(['id','item_sku_id']).unstack(0)['sale_qtty'].to_dict()
# orders_retail=defaultdict(lambda :defaultdict(int))
orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
# print orders_retail_new
# print orders_retail
for f in fdc:
    orders_retail[f]=defaultdict(lambda :defaultdict(int))
for k,v in orders_retail_mid.items():
    k1,k2=k.split(';')
    orders_retail[k1][k2]=dict(filter(lambda i:np.isnan(i[1])==False,v.items()))
# print orders_retail
#     for k1,v1 in v.items():
#         print k1,type(k1)
#         print v1,type(v1)
#订单类型:{订单id:类型}
orders_retail_type=defaultdict(str)
#sku当天从FDC的出库量，从RDC的出库量
sku_fdc_sales=defaultdict(int)
sku_rdc_sales=defaultdict(int)
#全量SKU列表
all_sku_list=list(set(allocation_sku_data['sku_id']))

###初始化仿真类，并运行相关结果
allocation=inventory_proess(fdc_forecast_sales,fdc_forecast_std,fdc_alt,fdc_alt_prob,fdc_inv,white_list_dict,fdc_allocation,fdc,rdc_inv,
                 order_list,date_range,orders_retail,all_sku_list)
allocation.OrdersSimulation()
#保持关键仿真数据
pickle.dump(fdc_forecast_sales,open(save_data_path+'fdc_forecast_sales','w'))
pickle.dump(fdc_forecast_std,open(save_data_path+'fdc_forecast_std','w'))
pickle.dump(fdc_alt,open(save_data_path+'fdc_alt','w'))
pickle.dump(fdc_alt_prob,open(save_data_path+'fdc_alt_prob','w'))
# pickle.dump(fdc_inv,open(save_data_path+'fdc_inv','w'))
# pickle.dump(white_list_dict,open(save_data_path+'white_list_dict','w'))
pickle.dump(fdc_allocation,open(save_data_path+'fdc_allocation','w'))
pickle.dump(rdc_inv,open(save_data_path+'rdc_inv','w'))
# pickle.dump(order_list,open(save_data_path+'order_list','w'))
# pickle.dump(orders_retail,open(save_data_path+'orders_retail','w'))
pickle.dump(all_sku_list,open(save_data_path+'all_sku_list','w'))
#####计算KPI，KPI主要包括本地订单满足率，周转，SKU满足率
#本地订单满足率 (本地出库订单+订单驱动内配)/订单数量
print 'origin orders......',allocation.orders_retail
print 'sim orders .......',allocation.simu_orders_retail
print 'fdc orders ......',allocation.fdc_simu_orders_retail
cnt_orders_retail_type={}
for k,v in allocation.orders_retail_type.items():
    cnt_orders_retail_type.setdefault(v,[]).append(k)
for k,v in cnt_orders_retail_type.items():
    print k,'has orders number:',len(v)
#周转，考核单个SKU的周转,考察一个SKU7天的周转，7天平均库存/7天的平均销量  订单数据：{fdc_订单时间_订单id:{SKU：数量}}
#将订单数据转换为{fdc{date：{sku,销量}}},同时需要判断订单是否有FDC出货，需要在仿真的过程中标记，便于后续获取计算
#直接标记不易标记，建立两个字典，一个记录仿真销量情况，一个记录仿真FDC销量情况
sale_orders_retail_sku_cnt=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
for f in fdc:
    for k,v in allocation.simu_orders_retail[f].items():
        date_sale=k[0:11]
        for k1,v1 in v.items():
            sale_orders_retail_sku_cnt[f][date_sale][k1]+=v1

# print allocation.fdc_inv[index]['inv'],将其拆解为{fdc:{date:{sku:inv}}}
inv_orders_retail_sku_cnt=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
for k,v in allocation.fdc_inv.items():
    print k
    k1,k2,k3=k.split('_')
    inv_orders_retail_sku_cnt[k1][k2][k3]=v['inv']

#遍历fdc,遍历日期，遍历sku,计算周转情况,ot_sku的数据格式：(fdc_sku_date:周转天数)
ot_sku=defaultdict(int)

for f in fdc:
    for i in len(date_range):
        sub_set=date_range[i:i+7]
        for sku in all_sku_list:
            v1=0
            v2=0
            for s in sub_set:
                v1+=sale_orders_retail_sku_cnt[f][s][sku]
                v2+=inv_orders_retail_sku_cnt[f][s][sku]
            index=gene_index(f,sku,date_range[i])
            ot_sku[index]=v2/v1
