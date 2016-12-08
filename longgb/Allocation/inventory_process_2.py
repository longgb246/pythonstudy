# coding=utf-8
from scipy.stats import rv_discrete,norm
import  numpy as np
import math
import  pandas as pd
from collections import defaultdict
import time,datetime
import copy


fdc_forecast_sales = 1
fdc_forecast_std = 1
fdc_alt = 1
fdc_alt_prob = 1
fdc_inv = 1
white_list_dict = 1
fdc_allocation = 1
fdc = 1
rdc_inv = 1
order_list = 1
date_range = 1
orders_retail = 1
all_sku_list = 1

white_list = 1
orders_retail_type = 1
simu_orders_retail = 1
fdc_simu_orders_retail = 1


def a__init__(fdc_forecast_sales,fdc_forecast_std,fdc_alt,fdc_alt_prob,fdc_inv,white_list_dict,fdc_allocation,fdc,rdc_inv,
             order_list,date_range,orders_retail,all_sku_list):
    ''' #类初始化函数，初始化类的各个参数'''
    #预测数据相关信息{fdc_sku_date:[7 days sales]},{fdc_sku_data:[7 days cv]}
    fdc_forecast_sales=fdc_forecast_sales
    fdc_forecast_std=fdc_forecast_std
    #RDC-->FDC时长分布,{fdc:[days]}}
    fdc_alt=fdc_alt
    fdc_alt_prob=fdc_alt_prob
    #defaultdict(lamda:defaultdict(int))
    fdc_inv=fdc_inv
    #白名单,不同日期的白名单不同{fdc:{date_s:[]}}
    white_list_dict=white_list_dict
    #调拨量字典,fdc_allocation=defaultdict(float)
    fdc_allocation=defaultdict(float)
    #fdc列表：
    fdc=fdc
    #RDC库存，{date_sku_rdc:库存量} defaultdict(int)
    rdc_inv=rdc_inv
    #订单数据，订单ID，SKU，实际到达量，到达时间,将其转换为{到达时间:{SKU：到达量}}形式的字典，defaultdict(lambda :defaultdict(int))
    order_list=order_list
    #仿真的时间窗口 时间格式如下：20161129
    date_range=date_range
    #订单数据：{fdc_订单时间_订单id:{SKU：数量}}
    orders_retail=orders_retail
    #记录仿真订单结果，存在订单，部分SKU不满足的情况
    simu_orders_retail=copy.deepcopy(orders_retail)
    #便于kpi计算，标记{fdc:{date:{sku:销量}}}
    fdc_simu_orders_retail=defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
    #订单类型:{订单id:类型}
    orders_retail_type=defaultdict(str)
    #sku当天从FDC的出库量，从RDC的出库量
    sku_fdc_sales=defaultdict(int)
    sku_rdc_sales=defaultdict(int)
    #全量SKU列表
    all_sku_list=all_sku_list

def gene_whitelist(date_s):
    # date_s = d
    # date_s = '20161104'
    '''获取该时间点的白名单'''
    white_list=defaultdict(list)
    for f in fdc:
        # f = fdc[1]
        # for i in white_list_dict[f]:
        date_s_time = time.strptime(date_s, '%Y%m%d')
        for i in white_list_dict.items():
            this_time = time.strptime(i[0], '%Y/%m/%d')
            # if i[0]<date_s:
            #     print i[0]
            if this_time <= date_s_time:
                # white_list[f].append(i[1][f])
                white_list[f].extend(i[1][f])

def cacl_rdc_inv(date_s):
    '''   #更新RDC库存,RDC库存的更新按照实际订单情况进行更新'''
    # date_s = d
    for s in all_sku_list:
        # s = all_sku_list[0]
        index=gene_index('rdc',s,date_s)
        # rdc_inv[index]=rdc_inv[index]+order_list[date_s][s]
        # order_list.keys()
        temp_date = time.strptime(date_s, '%Y%m%d')
        date_s_time = str(temp_date.tm_year)+'/'+str(temp_date.tm_mon)+'/'+str(temp_date.tm_mday)
        rdc_inv[index]=rdc_inv[index]+order_list[date_s_time][s]

def calc_lop(sku,fdc,date_s,cr=0.99):
    '''    #计算某个FDC的某个SKU的补货点'''
    #sku的FDC销量预测，与RDC的cv系数
    index=gene_index(fdc,sku,date_s)
    sku_sales=fdc_forecast_sales[index]
    sku_std= fdc_forecast_std[index]
    #具体的fdc对应的送货时长分布
    fdc_vlt=fdc_alt[fdc]
    fdc_vlt_porb=fdc_alt_prob[fdc]
    #计算送货期间的需求量
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

def calc_replacement(sku,fdc,date_s,sku_lop,bp=10,cr=0.99):
    '''
    #计算某个FDC的SKU的补货量
    计算补货量补货量为lop+bp-在途-当前库存
    '''
    #sku的FDC销量预测，与RDC的cv系数
    index=gene_index(fdc,sku,date_s)
    sku_sales=fdc_forecast_sales[index]
    sku_std= fdc_forecast_std[index]
    inv=fdc_inv[index]['inv']
    open_on=fdc_inv[index]['open_po']
    #计算BP长度内的消费量
    return sku_lop+sum(sku_sales[:(bp+1)])+\
    norm.ppf(cr)*math.sqrt(sum([i ** 2 for i in sku_std[:(bp+1)]]))-\
    inv-\
    open_on

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
    for s  in white_list:
        fdc_replacement=defaultdict(int)
        for f in fdc:
            lop_tmp=calc_lop(s,f,date_s)
            index=gene_index(f,s,date_s)
            if fdc_inv[index]['inv']<=lop_tmp:
                fdc_replacement[f]=calc_replacement(s,f,date_s)
            else:
                fdc_replacement[f]=0
        #rdc的索引应该是与日期相关的此处需要修改
        need_replacement=sum(fdc_replacement.values())
        if need_replacement>rdc_inv['sku']:
            #采用同比例缩放，亦可设置评判函数，采用贪心算法进行分类，可能存在非整数解，次数需要转换为整数解，待处理
            for f in fdc:
                fdc_replacement[f]=fdc_replacement[f]/need_replacement
        #更新调拨量，同时更新RDC库存
        for f in fdc:
            index=gene_index(f,s,date_s)
            fdc_allocation[index]=fdc_replacement[f]
            rdc_inv[index]['inv']=rdc_inv[index]['inv']-sum(fdc_replacement.values())

def gene_index(fdc,sku,date_s=''):
    '''
    #生成调用索引,将在多个地方调用该函数
    '''
    return str(date_s)+str(fdc)+str(sku)

def gene_alt(fdc):
    '''
    生成对应的调拨时长，用以更新FDC的库存
    '''
    fdc_vlt=fdc_alt[fdc]
    fdc_vlt_porb=fdc_alt_prob[fdc]
    alt_distribution = rv_discrete(values=(fdc_vlt, fdc_vlt_porb))
    return alt_distribution.rvs()

def calc_fdc_allocation(date_s,fdc):
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
    #计算补货点，判断补货量
    f = 1
    for s in white_list:        # 【这里有问题】
        # s1 = white_list[8]
        # s = s1[0]
        # index=gene_index(date_s,s,fdc)
        index=gene_index(f,s,date_s)
        #获取当前库存，当前库存已在订单循环部分完成
        #获取调拨量,从调拨字典中获取调拨量             # 【 fdc_inv 的日期有问题,索引有问题】
        fdc_inv[index]['inv']=fdc_inv[index]['inv']+fdc_inv[index]['arrive_quantity']
        fdc_inv[index]['allocation']=fdc_allocation[index]
        # alt=gene_alt(fdc)
        alt=gene_alt(f)                              # 【传参数不对，有问题】
        #更新在途量,c为标记变量
        c=0
        format_date='%Y%m%d'
        while c<alt:                                # 【这里的逻辑】
            date_tmp=datetime.datetime.strptime(date_s,format_date)+datetime.timedelta(c)
            date_s_c=date_tmp.strftime('%Y%m%d')
            index_tmp=gene_index(s,fdc,date_s_c)
            fdc_inv[index_tmp]['open_po']=fdc_inv[index_tmp]['open_po']+fdc_inv[index_tmp]['open_po']
        date_alt=datetime.datetime.strptime(date_s,format_date)+datetime.timedelta(alt)
        date_s_alt=date_alt.strftime(format_date)
        index_1=gene_index(s,fdc,date_s_alt)
        #更新当日到达量
        fdc_inv[index_1]['arrive_quantity']=fdc_inv[index]['allocation']+fdc_inv[index_1]['arrive_quantity']


def OrdersSimulation(self):
    for d in date_range:
        # d = date_range[0]
        # d = '20161101'
        #更新获取当天白名单`
        gene_whitelist(d)
        for f in fdc:
            # f = fdc[0]
            #增加RDC当天库存，并针对FDC进行调拨

            cacl_rdc_inv(d)

            calc_fdc_allocation(d,f)
            # for o in orders_retail[str(f)].items():  o = orders_retail[str(f)].items()[0]
            for o in orders_retail[f].items():
                #遍历订单,尽量按照时间顺序进行遍历
                #标记订单类型，第一位：1为FDC发货，0为内配驱动，9为RDC代发；第二位是否包含白名单 y包括白名单商品 n不包括白名单商品
                sku_state=[]
                for s in o[1].items():
                    # s = o[1].items()[0]
                    #遍历sku
                    index=gene_index(f,s[0],d)
                    index_rdc=gene_index('rdc',s[0],d)
                    tmp=defaultdict(int)
                    #如果sku不在白名单，则有RDC发货，RDC货不够发怎么办，这不科学啊
                    #
                    if s[0] not in white_list[f]:
                        #可以这么写  simu_orders_retail[index]=min(s[1],rdc_inv[index_rdc])
                        #但是不知道后期会不会增加什么标记，所以就先这么着吧
                        if s[1]>rdc_inv[index_rdc]:
                            #print 'what happened'
                            simu_orders_retail[index]=rdc_inv[index_rdc]
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                        else:
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                    #在白名单，但是fdc货不够发
                    elif s[1] > fdc_inv[index]['inv']:
                        #请求RDC协助，如果RDC不够，那么RDC+FDC应该够，如果不够，那也不科学啊
                        if s[1]>rdc_inv[index_rdc]:
                            if s[1]>rdc_inv[index_rdc]+fdc_inv[index]['inv']:
                                #print 'what happened'
                                simu_orders_retail[index]=rdc_inv[index_rdc]+fdc_inv[index]['inv']
                                fdc_simu_orders_retail[f][d][s[0]]=fdc_inv[index]['inv']
                                tmp['fdc_rdc']=1
                            else:
                                fdc_simu_orders_retail[f][d][s[0]]=fdc_inv[index]['inv']
                                tmp['fdc_rdc']=1
                        else:
                            tmp['rdc']=1;tmp['fdc_rdc']=1
                    #在白名单里面，货也够发
                    else:
                        fdc_simu_orders_retail[f][d][s[0]]=simu_orders_retail[index]
                        tmp['fdc']=1;tmp['fdc_rdc']=1
                        if s[1]<=rdc_inv[index_rdc]:
                            tmp['rdc']=1
                    sku_state.append(tmp)
                #标记订单类型,更新RDC库存，更新FDC库存
                flag_fdc=min([c['fdc'] for c in sku_state])
                flag_rdc=min([c['rdc'] for c in sku_state])
                flag_fdc_rdc=min([c['fdc_rdc'] for c in sku_state])
                if flag_fdc==1:
                    orders_retail_type[o[0]]='fdc'
                    for s in o[1].items():
                        index=gene_index(f,s,d)
                        fdc_inv[index]=fdc_inv[index]['inv']-min(s[1],fdc_inv[index]['inv'])
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
        #更新下一天库存，将当天剩余库存标记为第二天库存,第二天到达库存会在开始增加上
        for f in fdc:
            for s in white_list[f]:
                format_date='%Y%m%d'
                date_tmp=datetime.datetime.strptime(d,format_date)+datetime.timedelta(1)
                date_s_c=date_tmp.strftime('%Y%m%d')
                index_next=gene_index(s,f,date_s_c)
                index=gene_index(s,f,d)
                fdc_inv[index_next]['inv']=fdc_inv[index]['inv']
        #不仅仅更新白名单更新全量库存，如何获取全量list
        for s  in all_sku_list:
            index_next=gene_index(s,'rdc',d)
            index=gene_index(s,'rdc',d)
            rdc_inv[index_next]=rdc_inv[index]


