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
from inventory_process_batch import inventory_proess
import configServer
import  logging

#日志记录部分
# 创建一个logger
logger = logging.getLogger('allocation .. logger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(configServer.log_path)
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

##标记仿真的开始和结束日期
start_date='2016-10-01'#'2016-10-01'
end_date='2016-10-02'

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
#指定数据相关路径，数据加载分批加载清除操作，sale数据与sku数据按照三天的频次进行更新，即对应的字典中只保存三天的数据
#sku每次增加一天数据，开始放一天数据即可
sku_data_path=configServer.sku_data_path
#fdc放全量数据即可，此处为FDC的调拨时长分布
fdc_data_path=configServer.fdc_data_path
#FDC的初始化库存，只需要初始化库存即可
fdc_initialization_inv=configServer.fdc_initialization_inv
#调拨单放全量数据
order_data_path=configServer.order_data_path
#sale每次增加一天数据，开始放一天数据即可
sale_data_path=configServer.sale_data_path_batch
save_data_path=configServer.save_data_path#数据集存储路径
#读入相关数据集
#读入SKU粒度数据集
'''date,sku_id,dc_id,forecast_begin_date,forecast_days,forecast_daily_override_sales,
forecast_weekly_override_sales,forecast_weekly_std,inv,arrive_quantity,open_po,white_flag'''
# allocation_sku_data=pd.read_table(sku_data_path,sep='\001',
#                                   names=['sku_id','forecast_begin_date','forecast_days','forecast_daily_override_sales',
#                                          'forecast_weekly_override_sales','forecast_weekly_std','forecast_daily_std',
#                                          'variance','ofdsales','inv',
#                                          'arrive_quantity','open_po','white_flag'],header=None)
logger.info('开始读取sku数据并转化')
pkl_sku=open(sku_data_path)
allocation_sku_data=pickle.load(pkl_sku)
pkl_sku.close()
#print allocation_sku_data.head()
#allocation_sku_data.columns= ['sku_id','forecast_daily_override_sales','variance','ofdsales','inv','white_flag','white_flag_01','date_s','dc_id','variance_ofdsales','std']
allocation_sku_data.columns= ['sku_id','mean_sales','variance','ofdsales','inv','white_flag','white_flag_01','date_s','dc_id','variance_ofdsales','std']
# print allocation_sku_data.head() # update next step
logger.info('SKU数据读取完成')

# print allocation_sku_data.head(n=20)

#读入FDC粒度数据
'''
date,dc_id,alt,alt_prob
'''
# allocation_fdc_data=pd.read_table(fdc_data_path,sep='\001',
#                                 names=['org_from','org_to','actiontime_max','alt','alt_cnt'])
logger.info('开始读取fdc数据并转化')
pkl_fdc=open(fdc_data_path)
allocation_fdc_data=pickle.load(pkl_fdc)
pkl_fdc.close()
allocation_fdc_data.columns=['org_from','org_to','actiontime_max','alt','alt_cnt']
fdc_01=allocation_fdc_data.groupby(['org_from','org_to','alt']).sum()
fdc_01=fdc_01.reset_index()
fdc_02=allocation_fdc_data['alt_cnt'].groupby([allocation_fdc_data['org_from'],allocation_fdc_data['org_to']]).sum()
fdc_02=fdc_02.reset_index()
fdc_alt=pd.merge(fdc_01,fdc_02,on=['org_from','org_to'])
fdc_alt.columns=['org_from','org_to','alt','alt_cnt','alt_all_cnt']
fdc_alt['alt_prob']=fdc_alt['alt_cnt']/fdc_alt['alt_all_cnt']
allocation_fdc_data=fdc_alt
# allocation_fdc_data.columns=['org_from','dc_id','actiontime_max','alt','alt_cnt','alt_prob']
allocation_fdc_data.columns=['org_from','dc_id','alt','alt_cnt','alt_all_cnt','alt_prob']
allocation_fdc_data=allocation_fdc_data[allocation_fdc_data['org_from']==316]
#RDC-->FDC时长分布,{fdc:[days]}}
# fdc_alt_mid=pd.concat([allocation_fdc_data['dc_id'],
#                    allocation_fdc_data['alt']],axis=1)
# #print fdc_alt
# fdc_alt_mid.columns=['id','alt']
# fdc_alt=fdc_alt.set_index('id')['alt'].to_dict()
fdc_alt=defaultdict(list)
fdc_alt_prob=defaultdict(list)
for index,row in allocation_fdc_data.iterrows():
    if row['dc_id'] in fdc_alt:
        # print row['dc_id']
        # print  fdc_alt
        try:
            tmp=eval(row['alt'])
            fdc_alt[row['dc_id']].append(tmp)
        except:
            pass
    else:
        # print 'nooo'
        # print row['dc_id']
        # print  fdc_alt
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
# fdc_alt_prob_mid=pd.concat([allocation_fdc_data['dc_id'].astype('str'),
#                         allocation_fdc_data['alt_prob']],axis=1)
# fdc_alt_prob_mid.columns=['dc_id','alt_prob']

# for index,row in fdc_alt_prob_mid.iterrows():
#     if row['id'] in fdc_alt_prob:
#         fdc_alt_prob[row['id']]=fdc_alt_prob[row['id']].append(row['alt_prob'])
#     else:
#         fdc_alt_prob[row['id']]=[row['alt_prob']]
# fdc_alt_prob=fdc_alt_prob.set_index('id')['alt_prob'].to_dict()
# print fdc_alt
# print fdc_alt_prob





logger.info('fdc数据读取完成')
#读取fdc初始化数据，['sku_id','open_po_fdc','inv_fdc','date_s','dc_id']
logger.info('fdc初始化库存数据读取')
pkl_fdc_initialization=open(fdc_initialization_inv)
allocation_fdc_initialization=pickle.load(pkl_fdc_initialization)
pkl_sku.close()
allocation_fdc_initialization.columns=['sku_id','open_po_fdc','inv','date_s','dc_id']
logger.info('fdc初始化库存数据读取完成')
#读入采购单粒度数据
'''
date,rdc,order_id,item_sku_id,arrive_quantity
'''
# allocation_order_data=pd.read_table(order_data_path,sep='\001',
#                                   names=['arrive_time','item_sku_id','arrive_quantity'])
logger.info('开始读取order数据并转化')
pkl_order=open(order_data_path)
allocation_order_data=pickle.load(pkl_order)
pkl_order.close()
# print allocation_order_data.head()
allocation_order_data.columns=['arrive_time','item_sku_id','arrive_quantity','dc_id']
# allocation_order_data['arrive_quantity']=10
#print allocation_order_data.head()
logger.info('order数据读取完成')
#仿真的时间窗口 时间格式如下：2016-11-29
date_range=datelist(start_date,end_date)
logger.info('开始读取详单明细数据')
#读入订单明细数据
'''
date,sale_ord_id,item_sku_id,sale_qtty,sale_ord_tm,sale_ord_type,sale_ord_white_flag
'''
# allocation_sale_data=pd.read_csv(sale_data_path,sep='\001',
#                                  names=['org_dc_id','sale_ord_det_id','sale_ord_id','parent_sale_ord_id',
#                                         'item_sku_id','sale_qtty','sale_ord_tm','sale_ord_type','sale_ord_white_flag',
#
# pkl_sale=                                        'item_third_cate_c','item_second_cate_cd','date_s','shelves_dt,shelves_tm'])
pkl_sale=[]
for p in date_range:
    pkl_sale_mid=open(sale_data_path+p+'.pkl')
    mid_allocation_sale_data=pickle.load(pkl_sale_mid)
    pkl_sale.append(mid_allocation_sale_data)
    pkl_sale_mid.close()
allocation_sale_data=pd.concat(pkl_sale)
allocation_sale_data.columns=['org_dc_id', 'sale_ord_det_id', 'sale_ord_id', 'parent_sale_ord_id','item_sku_id',
                              'sale_qtty', 'sale_ord_tm', 'sale_ord_type', 'sale_ord_white_flag','white_flag_01', 'item_third_cate_cd',
                              'item_second_cate_cd', 'shelves_dt', 'shelves_tm', 'date_s', 'dc_id']
logger.info('详单明细数据读取完成')
#定义索引转换函数
def gene_index(self,fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return date_s+fdc+sku

##将上述读入的数据集，转换为调拨仿真类需要的数据集


# ######测试使用，增加临时分区数据##########
# allocation_sku_data['date']='2016-07-01'
# allocation_sku_data['dc_id']='628'
# allocation_sale_data['date']='2016-07-01'
# allocation_sale_data['dc_id']='628'
# allocation_fdc_data['date']='2016-07-01'
# allocation_fdc_data['dc_id']='628'
# allocation_order_data['date']='2016-07-01'
# allocation_order_data['dc_id']='628'
#预测数据相关信息{fdc_sku_date:[7 days sales]},{fdc_sku_data:[7 days cv]}
#该部分只考虑白名单的数据即可
logger.info('开始读取sku预测数据并转化')
# allocation_sku_data=allocation_sku_data[allocation_sku_data['white_flag']==1]
fdc_forecast_sales=pd.concat([allocation_sku_data['date_s'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                              +allocation_sku_data['sku_id'].astype('str'),
                              allocation_sku_data['mean_sales']],axis=1)
fdc_forecast_sales.columns=['id','forecast_value']
fdc_forecast_sales=fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()

# print fdc_forecast_sales

fdc_forecast_std=pd.concat([allocation_sku_data['date_s'].astype('str')+allocation_sku_data['dc_id'].astype('str')
                            +allocation_sku_data['sku_id'].astype('str'),
                            allocation_sku_data['std']],axis=1)
fdc_forecast_std.columns=['id','forecast_std']
fdc_forecast_std=fdc_forecast_std.set_index('id')['forecast_std'].to_dict()
logger.info('sku预测数据转化完成')
# #RDC-->FDC时长分布,{fdc:[days]}}
# fdc_alt=pd.concat([allocation_fdc_data['dc_id'],
#                    allocation_fdc_data['alt']],axis=1)
# #print fdc_alt
# fdc_alt.columns=['id','alt']
# fdc_alt=fdc_alt.set_index('id')['alt'].to_dict()
#
# fdc_alt_prob=pd.concat([allocation_fdc_data['dc_id'].astype('str'),
#                         allocation_fdc_data['alt_prob']],axis=1)
# fdc_alt_prob.columns=['id','alt_prob']
# fdc_alt_prob=fdc_alt_prob.set_index('id')['alt_prob'].to_dict()
# print fdc_alt
# print fdc_alt_prob

#defaultdict(lamda:defaultdict(int)),FDC只需要一个初始化库存即可,直接从FDC初始化库存中读取即可
fdc_inv=defaultdict(lambda :defaultdict(int))
# tmp_df=allocation_sku_data[allocation_sku_data['date_s']==start_date]
# mid_fdc_inv=pd.concat([tmp_df['date_s'].astype(str)+tmp_df['dc_id'].astype(str)+tmp_df['sku_id'].astype(str),
#                        tmp_df['inv']],axis=1)
# mid_fdc_inv.columns=['id','inv']
# mid_fdc_inv=mid_fdc_inv.drop_duplicates()
# mid_fdc_inv=mid_fdc_inv.set_index(0)['inv'].to_dict()
# tmp_df=pkl_fdc_initialization[pkl_fdc_initialization['date_s']==start_date]
allocation_fdc_initialization['inv']=allocation_fdc_initialization['inv']+allocation_fdc_initialization['open_po_fdc']
mid_fdc_inv=pd.concat([allocation_fdc_initialization['date_s'].astype(str)+allocation_fdc_initialization['dc_id'].astype(str)
                       +allocation_fdc_initialization['sku_id'].astype(str),
                       allocation_fdc_initialization['inv']],axis=1)
mid_fdc_inv.columns=['id','inv']
mid_fdc_inv=mid_fdc_inv.drop_duplicates()
# print mid_fdc_inv.head()
mid_fdc_inv=mid_fdc_inv.set_index('id')['inv'].to_dict()
for k,v in mid_fdc_inv.items():
    fdc_inv[k]['inv']=v
logger.info('开始生成白名单字典')
#白名单,不同日期的白名单不同{fdc:{date_s:[]}}
white_list_dict=defaultdict(lambda :defaultdict(list))
tmp_df=allocation_sku_data[allocation_sku_data['white_flag']==1][['date_s','sku_id','dc_id']]
for k,v in tmp_df['sku_id'].groupby([tmp_df['date_s'],tmp_df['dc_id']]):
    white_list_dict[k[1]][k[0]]=list(v)#
logger.info('白名单生成完成')
# print white_list_dict['']
#i调拨量字典,fdc_allocation=defaultdict(float)
fdc_allocation=''
#fdc列表：
#fdc=pd.unique(allocation_fdc_data['dc_id'])
fdc=['628','630','658']
#RDC库存，{date_sku_rdc:库存量} defaultdict(int),只需要初始化的RDC库存
rdc_inv=defaultdict(int)
# rdc_inv.update({i[0]:i[1] for i in fdc_inv.items() if 'rdc' in i[0] })
tmp_df=allocation_sku_data[allocation_sku_data['date_s']==start_date]
mid_rdc_inv=pd.concat([tmp_df['date_s'].astype(str)+'rdc'+tmp_df['sku_id'].astype(str),
                       tmp_df['inv']],axis=1)
mid_rdc_inv.columns=['id','inv']
mid_rdc_inv=mid_rdc_inv.drop_duplicates()
mid_rdc_inv=mid_rdc_inv.set_index('id')['inv'].to_dict()
rdc_inv.update(mid_rdc_inv)
#采购单数据，采购ID，SKU，实际到达量，到达时间,将其转换为{到达时间:{SKU：到达量}}形式的字典，defaultdict(lambda :defaultdict(int))
#采购单ID在这里没有作用，更关心的是单个SKU在某一天的到达量
#print allocation_order_data.head()
logger.info('开始处理采购单数据')
# print allocation_order_data.head()
tmp_df=allocation_order_data[['arrive_time','item_sku_id','arrive_quantity']]
# print 'tmp...', tmp_df.head()
tmp_df.columns=['date','item_sku_id','arrive_quantity']
order_list=defaultdict(lambda :defaultdict(int))
# tmp_df=tmp_df.drop_duplicates(subset=['date','item_sku_id'])
#tmp_dict=tmp_df.set_index(['date','item_sku_id']).unstack(0)['arrive_quantity'].to_dict()
# for index,row in tmp_df.iterrows():
#     print index,
#     print row['date']
#     print row['item_sku_id']
#     print row['arrive_quantity']
logger.info('进行字典推导更新')
# order_list.update({row['date']:{row['item_sku_id']:row['arrive_quantity']}
#                    for index,row in tmp_df.iterrows()})
for index,row in tmp_df.iterrows():
    if order_list.has_key(row['date']):
        if order_list[row['date']].has_key(row['item_sku_id']):
            order_list[row['date']][row['item_sku_id']]=order_list[row['date']][row['item_sku_id']]+row['arrive_quantity']
        else:
            order_list[row['date']][row['item_sku_id']]=row['arrive_quantity']
    else:
        order_list[row['date']]={row['item_sku_id']:row['arrive_quantity']}
logger.info('order字典更新完成')
logger.info('遍历中间字典，更新采购单字典')
# for k,v in tmp_dict.items():
#     #print k
#     #print '----------------------------------------'
#     for k1,v1 in v.items():
#         #print k1
#         #print '##########################################'
#         #print v1
#         if v1!=None:
#             order_list[k][k1]=v1
logger.info('采购单数据处理完成')
# print order_list




#订单数据：{fdc_订单时间_订单id:{SKU：数量}},当前的存储会造成的空间浪费应当剔除大小为0的SKU
logger.info('开始处理订单明细数据并转化')


tmp_df=allocation_sale_data[['dc_id','date_s','item_sku_id','sale_ord_id','sale_ord_tm','sale_qtty']]
tmp_df=pd.DataFrame(tmp_df)
orders_retail_mid=pd.concat([tmp_df['dc_id'].astype(str)+tmp_df['date_s'].astype(str),tmp_df['sale_ord_tm'].astype(str)+
                             tmp_df['sale_ord_id'].astype(str),tmp_df[['item_sku_id','sale_qtty']]],
                            axis=1)
orders_retail_mid.columns=['dc_date_id','id','item_sku_id','sale_qtty']
# orders_retail_mid=orders_retail_mid.drop_duplicates(subset=['id','item_sku_id'])

#orders_retail_mid=orders_retail_mid.set_index(['id','item_sku_id']).unstack(0)['sale_qtty'].to_dict()
# orders_retail=defaultdict(lambda :defaultdict(int))
# orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
orders_retail={}
for f in fdc:
   orders_retail[f]=defaultdict(lambda :defaultdict(int))
# orders_retail.update({row['dc_date_id']:{row['id']:{row['item_sku_id']:row['sale_qtty']}}
#                            for index,row in orders_retail_mid.iterrows()})
##能直接update 是因为订单编号是唯一的 所以这个key是唯一的 不会存在覆盖的现象
for index,row in orders_retail_mid.iterrows():
    if row['dc_date_id'] in orders_retail:
        orders_retail[row['dc_date_id']].update({row['id']:{row['item_sku_id']:row['sale_qtty']}})
    else:
        orders_retail[row['dc_date_id']]={row['id']:{row['item_sku_id']:row['sale_qtty']}}
# for index,row in orders_retail_mid.iterrows():
#     print row
# print orders_retail_new
# print orders_retail
# for k,v in orders_retail_mid.items():
#    k1,k2=k.split(';')
#    orders_retail[k1][k2]=dict(filter(lambda i:np.isnan(i[1])==False,v.items()))
# print orders_retail
#     for k1,v1 in v.items():
#         print k1,type(k1)
#         print v1,type(v1)
#订单类型:{订单id:类型}
logger.info('订单明细数据处理完成')
orders_retail_type=defaultdict(str)
#sku当天从FDC的出库量，从RDC的出库量
sku_fdc_sales=defaultdict(int)
sku_rdc_sales=defaultdict(int)
#全量SKU列表
all_sku_list=list(set(allocation_sku_data['sku_id'].values))


#print all_sku_list
logger.info('开始进行仿真运算')
###初始化仿真类，并运行相关结果
allocation=inventory_proess(fdc_forecast_sales,fdc_forecast_std,fdc_alt,fdc_alt_prob,fdc_inv,white_list_dict,fdc_allocation,fdc,rdc_inv,
                            order_list,date_range,orders_retail,all_sku_list,logger,save_data_path)
allocation.OrdersSimulation()
# allocation.save_result()
logger.info('仿真运算完成，开始进行数据保存与KPI计算')


#保持关键仿真数据
logger.info('开始保存仿真数据......')
pickle.dump(fdc_forecast_sales,open(save_data_path+'fdc_forecast_sales','w'))
pickle.dump(fdc_forecast_std,open(save_data_path+'fdc_forecast_std','w'))
pickle.dump(fdc_alt,open(save_data_path+'fdc_alt','w'))
pickle.dump(fdc_alt_prob,open(save_data_path+'fdc_alt_prob','w'))
pickle.dump(all_sku_list,open(save_data_path+'all_sku_list','w'))
pickle.dump(dict(allocation.fdc_inv),open(save_data_path+'fdc_inv','w'))
# pickle.dump(white_list_dict,open(save_data_path+'white_list_dict','w'))
pickle.dump(dict(allocation.fdc_allocation),open(save_data_path+'fdc_allocation','w'))
pickle.dump(dict(allocation.rdc_inv),open(save_data_path+'rdc_inv','w'))
# pickle.dump(dict(allocation.order_list),open(save_data_path+'order_list','w'))
# pickle.dump(dict(allocation.orders_retail),open(save_data_path+'orders_retail','w'))
# pickle.dump(dict(allocation.simu_orders_retail),open(save_data_path+'simu_orders_retail','w'))
# pickle.dump(dict(allocation.fdc_simu_orders_retail),open(save_data_path+'fdc_simu_orders_retail','w'))
####保持嵌套的字典#####
with open(save_data_path+'white_list_dict','w') as white:
    for k,v in white_list_dict.items():
        for k1,v1 in v.items():
            white.write(str(k))
            white.write('\t')
            white.write(str(k1))
            white.write('\t')
            white.write(str(v1))
        white.write('\n')
pickle.dump(dict(allocation.fdc_allocation),open(save_data_path+'fdc_allocation','w'))
pickle.dump(dict(allocation.rdc_inv),open(save_data_path+'rdc_inv','w'))

with open(save_data_path+'order_list','w') as ol:
    for k,v in allocation.order_list.items():
        for k1,v1 in v.items():
            ol.write(str(k))
            ol.write('\t')
            ol.write(str(k1))
            ol.write('\t')
            ol.write(str(v1))
        ol.write('\n')



with open(save_data_path+'orders_retail','w') as orl:
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
    with open(save_data_path+'simu_orders_retail','w') as orl:
        print allocation.simu_orders_retail.items()
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
    with open(save_data_path+'fdc_simu_orders_retail','w') as orl:
        for k,v in allocation.fdc_simu_orders_retail.items():
            print allocation.fdc_simu_orders_retail.items()
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
    print k
    k1,k2,k3=k[:11],k[11:14],k[14:] #仅针对三位数的FDC，如果采用其他的则需要考虑把FDC编码映射成三位或增加分隔符
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
print ot_sku