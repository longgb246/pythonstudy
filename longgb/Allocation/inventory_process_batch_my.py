# coding=utf-8
from scipy.stats import rv_discrete,norm
import  numpy as np
import math
import  pandas as pd
from collections import defaultdict,OrderedDict
import time,datetime
import copy
import pickle
import logging


# ================================================================================
# =                                 （1）开始仿真                                 =
# ================================================================================
# (1)dict：{'id'(date_s+dc_id+sku_id):'forecast_value'(mean_sales)}
fdc_forecast_sales = 1
fdc_forecast_sales=fdc_forecast_sales
# (2)dict：{'id'(date_s+dc_id+sku_id):'forecast_std'(std)}
fdc_forecast_std = 1
fdc_forecast_std=fdc_forecast_std
# (3)dict：{fdc:[alt]}
fdc_alt=defaultdict(list)
fdc_alt.update(fdc_alt)
# (4)dict：{fdc:[alt_prob]}
fdc_alt_prob=defaultdict(list)
fdc_alt_prob.update(fdc_alt_prob)
# (5)dict：{'id'(date_s+dc_id+sku_id):{'k':'inv'(inv)}}
fdc_inv = 1
fdc_inv=fdc_inv
# (6)dict：{fdc:{date_s:[]}}
white_list_dict = 1
white_list_dict=white_list_dict
# (7)RDC库存，{'id'(date_s+dc_id+sku_id):库存量}
rdc_inv=defaultdict(int)
rdc_inv.update(rdc_inv)
# (8)dict：{'date':{'sku':'arrive_quantity'}}
order_list = 1
order_list=order_list
# (9)dict：{'dc_date_id'(dc_id+date_s):{'id'(sale_ord_tm+sale_ord_id):{'item_sku_id':'sale_qtty'}}}
orders_retail = 1
orders_retail=orders_retail
# ============================= list =============================
# 仿真的时间窗口 时间格式如下：2016-11-29
date_range = 1
date_range=date_range
# fdc列表：
fdc = 1
fdc=fdc
# 全量SKU列表：
all_sku_list = 1
all_sku_list=all_sku_list
# ============================= 字段 =============================
logger = 1
logger=logger
save_data_path = 1
save_data_path=save_data_path
# ===============================================================
# =                           【新增】                          =
# ===============================================================
# (1)记录仿真订单结果，存在订单，部分SKU不满足的情况
simu_orders_retail=copy.deepcopy(orders_retail)
# (2)便于kpi计算，标记{fdc:{date:{sku:销量}}}
fdc_simu_orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
# (3)
fdc_allocation=defaultdict(float)
# (4)订单类型:{订单id:类型}
orders_retail_type=defaultdict(str)
# (5)sku当天从FDC的出库量，从RDC的出库量
sku_fdc_sales=defaultdict(int)
sku_rdc_sales=defaultdict(int)
# (6)list 总白名单 []
union_white_list = 1
# (7)dict：{fdc:[]}
white_list = 1


def gene_index(fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s)+str(fdc)+str(sku)


def gene_whitelist(date_s_func):
    '''获取该时间点的白名单,调用户一次刷新一次，只保存最新的白名单列表'''
    # date_s_func = d
    white_list=defaultdict(list)
    union_white_list=[]
    for f in fdc:
        for k,v in white_list_dict[f].items():
            if k==date_s_func:                  # SQL 里面已经处理好了，再回去看 SQL。
                white_list[f].extend(v)         # list[]
                union_white_list.extend(v)
        logger.info('当前日期--'+date_s_func+'当前FDC--'+f+'拥有的白名单数量为：'+str(len(white_list[f])))
    union_white_list=list(set(union_white_list))


def cacl_rdc_inv(date_s_func):
    '''  补货逻辑 #更新RDC库存,RDC库存的更新按照实际订单情况进行更新，rdc{index:库存量}'''
    # date_s_func = d
    for s in all_sku_list:
        # s = all_sku_list[0]
        if len(str(s))<3:
            continue
        index=gene_index('rdc',s,date_s_func)
        # rdc_inv {'id'(date_s + dc_id + sku_id): 库存量}
        # order_list {'date': {'sku': 'arrive_quantity'}}
        rdc_inv[index]=rdc_inv[index]+order_list[date_s_func].get(s,0)          # 【】


def calc_lop(sku_func,fdc_func,date_s,cr=0.99):
    '''    #计算某个FDC的某个SKU的补货点'''
    # sku_func = s
    # fdc_func = f
    # date_s = d
    # sku的FDC销量预测，与RDC的cv系数
    if sku_func not in union_white_list:
        return 0
    index=gene_index(fdc_func,sku_func,date_s)
    # dict：{'id'(date_s+dc_id+sku_id):'forecast_value'(mean_sales)}
    # sku_sales = [1, 1, 1, 1, 1, 1, 1]
    sku_sales=eval(fdc_forecast_sales[index])               # 【需要加异常处理吗？】
    try:
        sku_std= eval(fdc_forecast_std[index])
    except:
        sku_std=[0,0,0,0,0,0,0]
    # 默认将数据延展12周
    sku_sales=np.tile(sku_sales,12)                         # 【】
    sku_std=np.tile(sku_std,12)
    # 具体的fdc对应的送货时长分布
    # dict：{fdc: [alt]} fdc_func = 658
    fdc_vlt=fdc_alt[fdc_func]
    fdc_vlt_porb=fdc_alt_prob[fdc_func]
    if len(fdc_vlt)==0:
        fdc_vlt=[2]
        fdc_vlt_porb=[1]
    # 计算送货期间的需求量
    # 【没懂？在看！！！】
    demand_mean = [sum(sku_sales[:(l+1)]) for l in fdc_vlt]
    # VLT期间总销量均值的概率分布
    demand_mean_distribution = rv_discrete(values=(demand_mean, fdc_vlt_porb))
    part1 = demand_mean_distribution.mean()
    # 给定VLT，计算总销量的方差
    demand_var = [sum([i ** 2 for i in sku_std[:(l+1)]]) for l in fdc_vlt]
    # demand_std = np.sqrt(demand_var)
    # VLT期间总销量方差的概率分布
    demand_var_distribution = rv_discrete(values=(demand_var, fdc_vlt_porb))
    # 条件期望的方差
    part21 = demand_mean_distribution.var()
    # 条件方差的期望
    part22 = demand_var_distribution.mean()
    # 计算补货点
    lop = np.ceil(part1 + norm.ppf(cr) * math.sqrt(part21 + part22 + 0.1))
    return lop


def calc_replacement(sku_func,fdc_func,date_s,sku_lop,bp=10,cr=0.99):
    '''
    #计算某个FDC的SKU的补货量
    计算补货量补货量为 lop + bp - 在途 - 当前库存
    '''
    # sku_func = s
    # fdc_func = f
    # sku_lop = lop_tmp
    #sku的FDC销量预测，与RDC的cv系数
    index=gene_index(fdc_func,sku_func,date_s)
    sku_sales=eval(fdc_forecast_sales[index])           # 【异常诊断】
    try:
        sku_std= eval(fdc_forecast_std[index])
    except :
        sku_std=[0,0,0,0,0,0,0]
    inv=fdc_inv[index]['inv']
    open_on=fdc_inv[index]['open_po']
    #默认将数据延展12周
    sku_sales=np.tile(sku_sales,12)
    sku_std=np.tile(sku_std,12)
    #计算BP长度内的消费量
    return sku_lop+sum(sku_sales[:(bp+1)])+ \
           norm.ppf(cr)*math.sqrt(sum([i ** 2 for i in sku_std[:(bp+1)]])) - inv - open_on


def calc_sku_allocation(date_s):
    '''
    首先将当日到达量加到当日库存中
    @sku
    @调拨量
    ----输出---
    @date:日期
    @sku:sku
    @fdc:fdc
    @allocation：调拨量
    '''
    # date_s = d
    for s  in union_white_list:
        # s = union_white_list[0]
        fdc_replacement=defaultdict(int)
        for f in fdc:
            # f = fdc[0]
            if s not in white_list[f]:      # 是白名单，但不是这个fdc的白名单
                fdc_replacement[f]=0         # 补货量
            else:                           # 是白名单，也是这个fdc的白名单
                lop_tmp=calc_lop(s,f,date_s)    # lop_tmp = 7.0
                index=gene_index(f,s,date_s)
                # dict：{'id'(date_s + dc_id + sku_id): {'k': 'inv'(inv)}}
                if fdc_inv[index]['inv']<=lop_tmp:      # 补货量小于补货点，需要补货
                    fdc_replacement[f]=calc_replacement(s,f,date_s,lop_tmp)     # fdc_replacement[f] = 18.0
                else:
                    fdc_replacement[f]=0
        # rdc的索引应该是与日期相关的此处需要修改
        need_replacement=sum(fdc_replacement.values())      # 总需要的补货量
        index=gene_index('rdc',s,date_s)
        if need_replacement>rdc_inv[index]:
            # 采用同比例缩放，亦可设置评判函数，采用贪心算法进行分类，可能存在非整数解，次数需要转换为整数解，待处理
            tmp_inv_sum=0
            for f in fdc[:-1]:
                tmp=np.floor(fdc_replacement[f]/need_replacement*rdc_inv[index])
                fdc_replacement[f]=tmp
                tmp_inv_sum+=tmp
            fdc_replacement[fdc[-1]]=rdc_inv[index]-tmp_inv_sum
        # 更新调拨量，同时更新RDC库存
        for f in fdc:
            index=gene_index(f,s,date_s)
            fdc_allocation[index]=fdc_replacement[f]
            rdc_inv=gene_index('rdc',s,date_s)
            rdc_inv[index]=rdc_inv[index]-sum(fdc_replacement.values())


def gene_alt(fdc_func):
    '''
    生成对应的调拨时长，用以更新FDC的库存
    '''
    # fdc_func = f
    fdc_vlt=fdc_alt[fdc_func]
    fdc_vlt_porb=fdc_alt_prob[fdc_func]
    #如果没有对应的调拨时长，默认为3天
    if len(fdc_vlt)==0:
        return 3
    alt_distribution = rv_discrete(values=(fdc_vlt, fdc_vlt_porb))
    return alt_distribution.rvs()


def calc_fdc_allocation(date_s,fdc_func):
    '''
    ------输入数据格式----
    @date:日期 20161129,字符串格式
    @fdc:fdc
    ------输出数据格式,dataframe-----
    @date:日期
    @sku:sku
    @fdc:fdc
    @inv:当前库存
    @allocation：调拨量
    @open_po:在途量
    @arrive_quantity:当日到达量
    '''
    # date_s = d
    # fdc_func = f
    # 计算补货点，判断补货量
    # 遍历总白名单
    for s in union_white_list:
        index=gene_index(fdc_func,s,date_s)
        # 获取当前库存，当前库存已在订单循环部分完成
        # 获取调拨量,从调拨字典中获取调拨量
        # dict：{'id'(date_s+dc_id+sku_id):{'k':'inv'(inv)}}
        fdc_inv[index]['inv']=fdc_inv[index]['inv']+fdc_inv[index]['arrive_quantity']
        fdc_inv[index]['allocation']=fdc_allocation[index]
        alt=gene_alt(fdc_func)      # alt = 2.0
        # 更新在途量,c为标记变量
        c=0
        format_date='%Y-%m-%d'
        while c<alt:
            date_tmp=datetime.datetime.strptime(date_s,format_date)+datetime.timedelta(c)
            date_s_c=date_tmp.strftime('%Y-%m-%d')
            index_tmp=gene_index(fdc_func,s,date_s_c)
            fdc_inv[index_tmp]['open_po']=fdc_inv[index_tmp]['allocation']+fdc_inv[index_tmp]['open_po']        # 【这个地方有问题】
            c+=1
        date_alt=datetime.datetime.strptime(date_s,format_date)+datetime.timedelta(alt)
        date_s_alt=date_alt.strftime(format_date)
        index_1=gene_index(fdc_func,s,date_s_alt)
        # 更新当日到达量
        fdc_inv[index_1]['arrive_quantity']=fdc_inv[index]['allocation']+fdc_inv[index_1]['arrive_quantity']


def OrdersSimulation():
    # （1）循环遍历日期
    for d in date_range:
        # d = date_range[0]
        logger.info('begin to deal with '+d)
        logger.info('更新白名单信息')
        # （2.1）更新获取当天白名单
        gene_whitelist(d)
        logger.info('更新当天rdc库存')
        # （2.2）更新RDC库存
        cacl_rdc_inv(d)
        logger.info('计算每个SKU的调拨量')
        # （2.3）计算SKU的调拨量
        calc_sku_allocation(d)
        # （2）对 fdc 遍历
        for f in fdc:
            # f = fdc[0]
            #增加RDC当天库存，并针对FDC进行调拨
            logger.info('begin to deal with :'+d+'...fdc:'+f)
            # （3.1）计算fdc的调拨量和到达量
            calc_fdc_allocation(d,f)
            # dict：{'dc_date_id'(dc_id+date_s):{'id'(sale_ord_tm+sale_ord_id):{'item_sku_id':'sale_qtty'}}}
            tmp_order_retail=orders_retail[f+d]
            sorted_order_reatil=OrderedDict(sorted(tmp_order_retail.items(),key=lambda d:d[0]))
            print 'the number of retail ...of ',f,'..fdc..',len(sorted_order_reatil.items())
            # （3）遍历 某天+FDC 的 某订单id+销售时间 ： SKU 的 销售量. 遍历订单，按照时间顺序进行遍历
            for o in sorted_order_reatil.items():
                # o[0] = sorted_order_reatil.keys()[0]
                # o[1] = sorted_order_reatil[sorted_order_reatil.keys()[0]]
                # o.append(sorted_order_reatil.keys()[0])
                # o.append(sorted_order_reatil[sorted_order_reatil.keys()[0]])
                # 标记订单类型，第一位：1为FDC发货，0为内配驱动，9为RDC代发；第二位是否包含白名单 y包括白名单商品 n不包括白名单商品
                # logger.info('该订单信息如下：...')
                order_index=o[0]
                sku_state=[]
                # （4）遍历 SKU：销售量
                for s in o[1].items():
                    # s[0] = o[1].keys()
                    # s[1] = o[1].values()
                    # s = []
                    # s.append(o[1].keys()[0])
                    # s.append(o[1].values()[0])
                    index=gene_index(f,s[0],d)
                    index_rdc=gene_index('rdc',s[0],d)
                    tmp=defaultdict(int)
                    # 如果sku不在白名单，则有RDC发货，RDC货不够发怎么办，这不科学啊
                    if s[0] not in white_list[f]:
                        # 可以这么写  simu_orders_retail[index]=min(s[1],rdc_inv[index_rdc])
                        # 但是不知道后期会不会增加什么标记，所以就先这么着吧
                        if s[1]>rdc_inv[index_rdc]:
                            simu_orders_retail[f+d][order_index]=rdc_inv[index_rdc]
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                        else:
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                    # 在白名单，但是fdc货不够发
                    elif s[1] > fdc_inv[index]['inv']:
                        # print s[0],' 属于白名单,但是fdc的库存不够卖的'
                        # 请求RDC协助，如果RDC不够，那么RDC+FDC应该够，如果不够，那也不科学啊
                        if s[1]>rdc_inv[index_rdc]:
                            # print s[0],' 属于白名单,但是rdc的库存也不够卖的'
                            if s[1]>rdc_inv[index_rdc]+fdc_inv[index]['inv']:
                                simu_orders_retail[f+d][order_index]=rdc_inv[index_rdc]+fdc_inv[index]['inv']
                                fdc_simu_orders_retail[f+d][order_index][s[0]]=fdc_inv[index]['inv']
                                tmp['fdc_rdc']=1
                            else:
                                # print s[0],' 属于白名单,但是rdc+fdc的库存够卖的'
                                fdc_simu_orders_retail[f+d][order_index][s[0]]=fdc_inv[index]['inv']
                                tmp['fdc_rdc']=1
                        else:
                            # print s[0],' 属于白名单,但是rdc的库存够卖的'
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                    #在白名单里面，货也够发
                    else:
                        # print s[0],' 属于白名单,但是fdc的库存够卖的'
                        fdc_simu_orders_retail[f+d][order_index][s[0]]=simu_orders_retail[f+d][order_index][s[0]]
                        tmp['fdc']=1;tmp['fdc_rdc']=1
                        if s[1]<=rdc_inv[index_rdc]:
                            tmp['rdc']=1
                    sku_state.append(tmp)
                # 标记订单类型,更新RDC库存，更新FDC库存
                flag_fdc=min([c['fdc'] for c in sku_state])
                flag_rdc=min([c['rdc'] for c in sku_state])
                flag_fdc_rdc=min([c['fdc_rdc'] for c in sku_state])
                if flag_fdc==1:
                    orders_retail_type[o[0]]='fdc'
                    for s in o[1].items():
                        index=gene_index(f,s,d)
                        fdc_inv[index]['inv']=fdc_inv[index]['inv']-min(s[1],fdc_inv[index]['inv'])
                elif flag_rdc==1:
                    orders_retail_type[o[0]]='rdc'
                    for s in o[1].items():
                        index=gene_index(f,s,d)
                        rdc_inv[index]=rdc_inv[index]-min(s[1],rdc_inv[index])
                elif flag_fdc_rdc==1:
                    orders_retail_type[o[0]]='fdc_rdc'
                    for s in o[1].items():
                        index=gene_index(f,s,d)
                        sku_gap=s[1]-fdc_inv[index]['inv']
                        fdc_inv[index]['inv']=0 if sku_gap>=0 else sku_gap
                        rdc_inv[index]=rdc_inv[index] if sku_gap<0 else rdc_inv[index]-sku_gap
                else:
                    pass
        # （5）对 fdc 遍历
        # 更新下一天库存，将当天剩余库存标记为第二天库存,第二天到达库存会在开始增加上
        for f in fdc:
            for s in white_list[f]:
                format_date='%Y-%m-%d'
                date_tmp=datetime.datetime.strptime(d,format_date)+datetime.timedelta(1)
                date_s_c=date_tmp.strftime('%Y-%m-%d')
                index_next=gene_index(f,s,date_s_c)
                index=gene_index(f,s,d)
                fdc_inv[index_next]['inv']=fdc_inv[index]['inv']
        # 不仅仅更新白名单更新全量库存，如何获取全量list
        # （6）对 all_sku_list 遍历, 更新 rdc 库存
        for s in all_sku_list:
            index_next=gene_index('rdc',s,d)
            index=gene_index('rdc',s,d)
            rdc_inv[index_next]=rdc_inv[index]
            # 此处对 预测数据和明细数据进行更新fdc_forecast_std,fdc_forecast_sales,orders_retail
            # 删除已经使用的数据
        # （7）删除？
        for f in fdc:
            for s in all_sku_list:
                index_del=gene_index(f,s,d)
                if fdc_forecast_sales.has_key(index_del):
                    fdc_forecast_sales[index_del]
                if fdc_forecast_sales.has_key(index_del):
                    fdc_forecast_std[index_del]
        del_orders_retail_list=[]
        for k,v in orders_retail.items():
            if d in str(k):
                del_orders_retail_list.append(k)
        for k in del_orders_retail_list:
            del orders_retail[k]
        # （8）增加下一天的数据：预测的数据。SKU 表
        logger.info('update next day datas')
        start_date = datetime.datetime.strptime(d,'%Y-%m-%d')+datetime.timedelta(1)
        start_date=datetime.datetime.strftime(start_date,'%Y-%m-%d')
        if start_date not in date_range:
            continue
        tmp_sku_path='/home/cmo_ipc/Allocation_shell/datasets/data_total3/total_sku/'+start_date+'.pkl'
        pkl_sale=open(tmp_sku_path)
        sku_day_data=pickle.load(pkl_sale)
        pkl_sale.close()
        # sku_day_data=sku_day_data[sku_day_data['white_flag']==1]
        tmp_fdc_forecast_sales=pd.concat([sku_day_data['date_s'].astype('str')+sku_day_data['dc_id'].astype('str')
                                          +sku_day_data['sku_id'].astype('str'),
                                          sku_day_data['forecast_daily_override_sales']],axis=1)
        tmp_fdc_forecast_sales.columns=['id','forecast_value']
        tmp_fdc_forecast_sales=tmp_fdc_forecast_sales.set_index('id')['forecast_value'].to_dict()
        fdc_forecast_sales.update(copy.deepcopy(tmp_fdc_forecast_sales))
        tmp_fdc_forecast_std=pd.concat([sku_day_data['date_s'].astype('str')+sku_day_data['dc_id'].astype('str')
                                        +sku_day_data['sku_id'].astype('str'),
                                        sku_day_data['std']],axis=1)
        tmp_fdc_forecast_std.columns=['id','forecast_std']
        tmp_fdc_forecast_std=tmp_fdc_forecast_std.set_index('id')['forecast_std'].to_dict()
        fdc_forecast_std.update(copy.deepcopy(tmp_fdc_forecast_std))
        # （9）更白名单
        # white_list_dict=defaultdict(lambda :defaultdict(list))
        tmp_df=sku_day_data[sku_day_data['white_flag']==1][['date_s','sku_id','dc_id']]
        for k,v in tmp_df['sku_id'].groupby([tmp_df['date_s'],tmp_df['dc_id']]):
            # print k[1],k[0]
            white_list_dict[k[1]][k[0]]=list(v)#
        logger.info('日sku数据更新完成')

