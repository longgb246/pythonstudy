# coding=utf-8
from scipy.stats import rv_discrete, norm
import numpy as np
import math
import pandas as pd
from collections import defaultdict, OrderedDict
import time, datetime
import copy
import pickle
import logging
import os


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


class inventory_proess:
    def __init__(self, fdc_safe, fdc_max, fdc_alt, fdc_alt_prob, fdc_inv, white_list_dict,
            fdc_allocation, fdc, rdc_inv,order_list, date_range, orders_retail, all_sku_list, logger, save_data_path):
        ''' #类初始化函数，初始化类的各个参数'''
        # 预测数据相关信息{fdc_sku_date:[7 days sales]},{fdc_sku_data:[7 days cv]}
        self.fdc_safe = fdc_safe
        self.fdc_max = fdc_max
        # RDC-->FDC时长分布,{fdc:[days]}}
        self.fdc_alt = defaultdict(list)
        self.fdc_alt.update(fdc_alt)
        self.fdc_alt_prob = defaultdict(list)
        self.fdc_alt_prob.update(fdc_alt_prob)
        # defaultdict(lamda:defaultdict(int)),包括库存，调拨量，在途，到达量
        self.fdc_inv = fdc_inv
        # 白名单,不同日期的白名单不同{fdc:{date_s:[]}}
        self.white_list_dict = white_list_dict
        # 调拨量字典,fdc_allocation=defaultdict(float)
        self.fdc_allocation = defaultdict(float)
        # fdc列表：
        self.fdc = fdc
        # RDC库存，{date_sku_rdc:库存量} defaultdict(int)
        self.rdc_inv = defaultdict(int)
        self.rdc_inv.update(rdc_inv)
        # 订单数据，订单ID，SKU，实际到达量，到达时间,将其转换为{到达时间:{SKU：到达量}}形式的字典，defaultdict(lambda :defaultdict(int))
        self.order_list = order_list
        # 仿真的时间窗口 时间格式如下：20161129
        self.date_range = date_range
        # 订单数据：{fdc_订单时间_订单id:{SKU：数量}}
        self.orders_retail = orders_retail
        # 记录仿真订单结果，存在订单，部分SKU不满足的情况
        self.simu_orders_retail = copy.deepcopy(self.orders_retail)
        # 便于kpi计算，标记{fdc:{date:{sku:销量}}}
        self.fdc_simu_orders_retail = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # 订单类型:{订单id:类型}
        self.orders_retail_type = defaultdict(str)
        # sku当天从FDC的出库量，从RDC的出库量
        self.sku_fdc_sales = defaultdict(int)
        self.sku_rdc_sales = defaultdict(int)
        # 全量SKU列表
        self.all_sku_list = all_sku_list
        self.logger = logger
        self.save_data_path = save_data_path

    def gene_whitelist(self, date_s):
        '''获取该时间点的白名单,调用户一次刷新一次，只保存最新的白名单列表'''
        self.white_list = defaultdict(list)
        self.union_white_list = []
        for f in self.fdc:
            for k, v in self.white_list_dict[f].items():
                if k == date_s:
                    self.white_list[f].extend(v)  # list[]
                    self.union_white_list.extend(v)
            self.logger.info('当前日期--' + date_s + '当前FDC--' + f + '拥有的白名单数量为：' + str(len(self.white_list[f])))
        self.union_white_list = list(set(self.union_white_list))

    def cacl_rdc_inv(self, date_s):
        '''  补货逻辑 #更新RDC库存,RDC库存的更新按照实际订单情况进行更新，rdc{index:库存量}'''
        for s in self.all_sku_list:
            if len(str(s)) < 3:
                continue
            index = self.gene_index('rdc', s, date_s)
            self.rdc_inv[index] = self.rdc_inv[index] + self.order_list[date_s].get(s, 0)

    def calc_lop(self, sku, fdc, date_s, cr=0.99):
        '''    #计算某个FDC的某个SKU的补货点'''
        # sku的FDC销量预测，与RDC的cv系数
        if sku not in self.union_white_list:
            return 0
        index = self.gene_index(fdc, sku, date_s)
        sku_sales = eval(self.fdc_safe[index])
        try:
            sku_std = eval(self.fdc_max[index])
        except:
            sku_std = [0, 0, 0, 0, 0, 0, 0]
        sku_sales_mean = np.mean(sku_sales)
        safe_qtty = sku_sales_mean * 4
        lop = safe_qtty
        # # # 默认将数据延展12周
        # sku_sales = np.tile(sku_sales, 12)
        # sku_std = np.tile(sku_std, 12)
        # # 具体的fdc对应的送货时长分布
        # fdc_vlt = self.fdc_alt[fdc]
        # fdc_vlt_porb = self.fdc_alt_prob[fdc]
        # if len(fdc_vlt) == 0:
        #     fdc_vlt = [2]
        #     fdc_vlt_porb = [1]
        # demand_mean = [sum(sku_sales[:(l + 1)]) for l in fdc_vlt]
        # # VLT期间总销量均值的概率分布
        # demand_mean_distribution = rv_discrete(values=(demand_mean, fdc_vlt_porb))
        # part1 = demand_mean_distribution.mean()
        # # 给定VLT，计算总销量的方差
        # demand_var = [sum([i ** 2 for i in sku_std[:(l + 1)]]) for l in fdc_vlt]
        # # demand_std = np.sqrt(demand_var)
        # # VLT期间总销量方差的概率分布
        # demand_var_distribution = rv_discrete(values=(demand_var, fdc_vlt_porb))
        # # 条件期望的方差
        # part21 = demand_mean_distribution.var()
        # # 条件方差的期望
        # part22 = demand_var_distribution.mean()
        # # 计算补货点
        # lop = np.ceil(part1 + norm.ppf(cr) * math.sqrt(part21 + part22 + 0.1))
        return lop

    def calc_replacement(self, sku, fdc, date_s, sku_lop, bp=10, cr=0.99):
        '''
        #计算某个FDC的SKU的补货量
        计算补货量补货量为lop+bp-在途-当前库存
        '''
        # sku的FDC销量预测，与RDC的cv系数
        index = self.gene_index(fdc, sku, date_s)
        max_qtty = self.fdc_max[index]
        inv = self.fdc_inv[index]['inv']
        open_on = self.fdc_inv[index]['open_po']
        lop_replacement = max(max_qtty - inv - open_on,0)
        # 调整补货量
        if lop_replacement <= 10:
            pass
        else:
            div_num, mod_num = divmod(lop_replacement, 10)
            if mod_num <= 2:
                lop_replacement = div_num * 10
            elif mod_num <= 7:
                lop_replacement = div_num * 10 + 5
            else:
                lop_replacement = (div_num + 1) * 10
        # # 默认将数据延展12周
        # sku_sales = np.tile(sku_sales, 12)
        # sku_std = np.tile(sku_std, 12)
        # # 计算BP长度内的消费量
        # lop_replacement = max(np.ceil(sku_lop + sum(sku_sales[:(bp + 1)]) +
        #                 norm.ppf(cr) * math.sqrt(sum([i ** 2 for i in sku_std[:(bp + 1)]])) - inv - open_on), 0)
        return np.floor(lop_replacement)

    def calc_sku_allocation(self, date_s):
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
        for s in self.union_white_list:
            fdc_replacement = defaultdict(int)
            for f in self.fdc:
                if s not in self.white_list[f]:
                    fdc_replacement[f] = 0
                else:
                    index = self.gene_index(f, s, date_s)
                    lop_tmp = self.fdc_safe[index]
                    # 在途加上库存低于补货点
                    if (self.fdc_inv[index]['inv'] + self.fdc_inv[index]['open_po']) < lop_tmp:
                        fdc_replacement[f] = self.calc_replacement(s, f, date_s, lop_tmp)
                    else:
                        fdc_replacement[f] = 0
            index_rdc = self.gene_index('rdc', s, date_s)
            rdc_inv_avail = max(np.min([self.rdc_inv[index_rdc] - 12, np.floor(self.rdc_inv[index_rdc] * 0.2)]),0)
            # 更新实际调拨
            for f in ['630', '658', '628']:
                index_fdc = self.gene_index(f, s, date_s)
                fdc_inv_avail = np.min([fdc_replacement[f], rdc_inv_avail])
                self.fdc_allocation[index_fdc] = fdc_inv_avail
                self.rdc_inv[index_rdc] = self.rdc_inv[index_rdc] - fdc_inv_avail
                rdc_inv_avail -= fdc_inv_avail

    def gene_index(self, fdc, sku, date_s=''):
        '''
        #生成调用索引,将在多个地方调用该函数
        '''
        return str(date_s) + str(fdc) + str(sku)

    def gene_alt(self, fdc):
        '''
        生成对应的调拨时长，用以更新FDC的库存
        '''
        fdc_vlt = self.fdc_alt[fdc]
        fdc_vlt_porb = self.fdc_alt_prob[fdc]
        # 如果没有对应的调拨时长，默认为3天
        if len(fdc_vlt) == 0:
            return 3
        alt_distribution = rv_discrete(values=(fdc_vlt, fdc_vlt_porb))
        return alt_distribution.rvs()

    def calc_fdc_allocation(self, date_s, fdc):
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
        # 计算补货点，判断补货量
        # 假设一个RDC-FDC同一天的调拨 运达实际相同
        for s in self.union_white_list:
            index = self.gene_index(fdc, s, date_s)
            # 获取当前库存，当前库存已在订单循环部分完成
            # 获取调拨量,从调拨字典中获取调拨量
            if self.fdc_allocation[index] > 0:
                self.fdc_inv[index]['allocation'] = self.fdc_allocation[index]
                # 放在这里，因为同一个调拨单的也不能保证同一天到达，所以按照SKU进行时长抽样
                alt = self.gene_alt(fdc)
                # 更新在途量,c为标记变量
                c = 0
                format_date = '%Y-%m-%d'
                while c < alt:
                    date_tmp = datetime.datetime.strptime(date_s, format_date) + datetime.timedelta(c)
                    date_s_c = date_tmp.strftime('%Y-%m-%d')
                    index_tmp = self.gene_index(fdc, s, date_s_c)
                    self.fdc_inv[index_tmp]['open_po'] = self.fdc_inv[index]['allocation'] + self.fdc_inv[index_tmp]['open_po']
                    c += 1
                date_alt = datetime.datetime.strptime(date_s, format_date) + datetime.timedelta(alt)
                date_s_alt = date_alt.strftime(format_date)
                index_1 = self.gene_index(fdc, s, date_s_alt)
                # 更新当日到达量
                self.fdc_inv[index_1]['arrive_quantity'] = self.fdc_inv[index]['allocation'] + self.fdc_inv[index_1]['arrive_quantity']
            self.fdc_inv[index]['inv'] = self.fdc_inv[index]['inv'] + self.fdc_inv[index]['arrive_quantity']

    def mkdir_save(self):
        self.save_data_path_org = self.save_data_path + 'org_data'
        if os.path.exists(self.save_data_path_org) == False:
            os.mkdir(self.save_data_path_org)
        self.save_data_path_fdc_inv = self.save_data_path + 'fdc_inv'
        if os.path.exists(self.save_data_path_fdc_inv) == False:
            os.mkdir(self.save_data_path_fdc_inv)
        self.save_data_path_fdc_allocation = self.save_data_path + 'fdc_allocation'
        if os.path.exists(self.save_data_path_fdc_allocation) == False:
            os.mkdir(self.save_data_path_fdc_allocation)
        self.save_data_path_rdc_inv = self.save_data_path + 'rdc_inv'
        if os.path.exists(self.save_data_path_rdc_inv) == False:
            os.mkdir(self.save_data_path_rdc_inv)
        self.save_data_path_white_list = self.save_data_path + 'white_list'
        if os.path.exists(self.save_data_path_white_list) == False:
            os.mkdir(self.save_data_path_white_list)
        self.save_data_path_order_list = self.save_data_path + 'order_list'
        if os.path.exists(self.save_data_path_order_list) == False:
            os.mkdir(self.save_data_path_order_list)
        self.save_data_path_white_list = self.save_data_path + 'white_list'
        if os.path.exists(self.save_data_path_white_list) == False:
            os.mkdir(self.save_data_path_white_list)
        self.save_data_path_orders_retail = self.save_data_path + 'orders_retail'
        if os.path.exists(self.save_data_path_orders_retail) == False:
            os.mkdir(self.save_data_path_orders_retail)
        self.save_data_path_simu_orders_retail = self.save_data_path + 'simu_orders_retail'
        if os.path.exists(self.save_data_path_simu_orders_retail) == False:
            os.mkdir(self.save_data_path_simu_orders_retail)
        self.save_data_path_fdc_simu_orders_retail = self.save_data_path + 'fdc_simu_orders_retail'
        if os.path.exists(self.save_data_path_fdc_simu_orders_retail) == False:
            os.mkdir(self.save_data_path_fdc_simu_orders_retail)

    def save_oneday(self, date_s):
        self.logger.info('Save the median Data : {0} ...'.format(date_s))
        pickle.dump(dict(self.fdc_inv),
                    open(self.save_data_path_fdc_inv + os.sep + 'fdc_inv_{0}.pkl'.format(date_s), 'w'))
        pickle.dump(dict(self.fdc_allocation),
                    open(self.save_data_path_fdc_allocation + os.sep + 'fdc_allocation_{0}.pkl'.format(date_s), 'w'))
        pickle.dump(dict(self.rdc_inv),
                    open(self.save_data_path_rdc_inv + os.sep + 'rdc_inv_{0}.pkl'.format(date_s), 'w'))
        with open(self.save_data_path_order_list + os.sep + 'order_list_{0}.txt'.format(date_s), 'w') as ol:
            for k, v in self.order_list.items():
                for k1, v1 in v.items():
                    ol.write(str(k))
                    ol.write('\t')
                    ol.write(str(k1))
                    ol.write('\t')
                    ol.write(str(v1))
                ol.write('\n')
        with open(self.save_data_path_orders_retail + os.sep + 'orders_retail_{0}.txt'.format(date_s), 'w') as orl:
            for k, v in self.orders_retail.items():
                for k1, v1 in v.items():
                    for k2, v2 in v1.items():
                        orl.write(str(k))
                        orl.write('\t')
                        orl.write(str(k1))
                        orl.write('\t')
                        orl.write(str(k2))
                        orl.write('\t')
                        orl.write(str(v2))
                orl.write('\n')
        try:
            with open(self.save_data_path_simu_orders_retail + os.sep + 'simu_orders_retail_{0}.txt'.format(date_s),
                      'w') as orl:
                for k, v in self.simu_orders_retail.items():
                    for k1, v1 in v.items():
                        for k2, v2 in v1.items():
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
            with open(self.save_data_path_fdc_simu_orders_retail + os.sep + 'fdc_simu_orders_retail_{0}.txt'.format(
                    date_s), 'w') as orl:
                for k, v in self.fdc_simu_orders_retail.items():
                    for k1, v1 in v.items():
                        for k2, v2 in v1.items():
                            orl.write(str(k))
                            orl.write('\t')
                            orl.write(str(k1))
                            orl.write('\t')
                            orl.write(str(k2))
                            orl.write('\t')
                            orl.write(str(v2))
                    orl.write('\n')
        except:
            print 'in the except'
        self.logger.info('Save the median Data : {0} . Finish !'.format(date_s))

    def OrdersSimulation(self):
        # 分时段存储
        # save_date = ["2016-10-05", "2016-10-10", "2016-10-15", "2016-10-20", "2016-10-25"]
        # self.mkdir_save()
        for d in self.date_range:
            # 1.1 更新获取当天白名单
            self.logger.info('begin to deal with ' + d)
            self.logger.info('更新白名单信息')
            t1 = time.time()
            self.gene_whitelist(d)
            printruntime(t1, 'gene_whitelist')
            # 1.2 更新RDC库存，增
            self.logger.info('更新当天rdc库存')
            t1 = time.time()
            self.cacl_rdc_inv(d)
            printruntime(t1, 'cacl_rdc_inv')
            # 1.3 计算每个SKU的调拨量
            self.logger.info('计算每个SKU的调拨量')
            t1 = time.time()
            self.calc_sku_allocation(d)
            printruntime(t1, 'calc_sku_allocation')
            all_rdc_orders_reatail = {}
            top_n_min_orders_retail = {}
            # 1.4 未知
            t1 = time.time()
            for f in self.fdc:
                # 更新FDC当天库存，增
                # 并针对FDC进行调拨，仅更新调拨、在途、到达，并未减
                self.logger.info('begin to deal with :' + d + '...fdc:' + f)
                self.calc_fdc_allocation(d, f)
                tmp_order_retail = self.orders_retail[f + d]
                all_rdc_orders_reatail[f] = copy.deepcopy(sorted(tmp_order_retail.items(), key=lambda d: d[0]))
                top_n_min_orders_retail[f] = all_rdc_orders_reatail[f].pop(0)
            printruntime(t1, 'calc_fdc_allocation')
            t1 = time.time()
            while top_n_min_orders_retail:
                o = copy.deepcopy(sorted(top_n_min_orders_retail.items(), key=lambda t: t[1][0]).pop(0))
                f = o[0]
                del top_n_min_orders_retail[f]
                order_index = o[1][0]
                sku_state = []
                for s in dict(o[1][1]).items():
                    # 遍历sku
                    index = self.gene_index(f, s[0], d)
                    index_rdc = self.gene_index('rdc', s[0], d)
                    tmp = defaultdict(int)
                    if s[0] not in self.white_list[f]:
                        # print s[0],' 不属于白名单'
                        if s[1] >= self.rdc_inv[index_rdc] >= 0:
                            # print 'what happened'
                            # self.simu_orders_retail[f + d][order_index][s[0]] = self.rdc_inv[index_rdc]
                            tmp['rdc'] = 1
                            tmp['fdc_rdc'] = 1
                        else:
                            # 该订单对应的sku满足标记对应的类型
                            tmp['rdc'] = 1
                            tmp['fdc_rdc'] = 1
                            # 在白名单，但是fdc货不够发
                    elif s[1] > self.fdc_inv[index]['inv']:
                        # print s[0],' 属于白名单,但是fdc的库存不够卖的'
                        # 请求RDC协助，如果RDC不够，那么RDC+FDC应该够，如果不够，那也不科学啊
                        if s[1] > self.rdc_inv[index_rdc]:
                            # print s[0],' 属于白名单,但是rdc的库存也不够卖的'
                            if s[1] > self.rdc_inv[index_rdc] + self.fdc_inv[index]['inv']:
                                # print 'what happened'
                                # self.simu_orders_retail[f + d][order_index][s[0]] = self.rdc_inv[index_rdc] + self.fdc_inv[index]['inv']
                                # self.fdc_simu_orders_retail[f + d][order_index][s[0]] = self.fdc_inv[index]['inv']
                                tmp['fdc_rdc'] = 1
                            else:
                                # print s[0],' 属于白名单,但是rdc+fdc的库存够卖的'
                                # self.fdc_simu_orders_retail[f + d][order_index][s[0]] = self.fdc_inv[index]['inv']
                                tmp['fdc_rdc'] = 1
                        else:
                            # print s[0],' 属于白名单,但是rdc的库存够卖的'
                            tmp['rdc'] = 1
                            tmp['fdc_rdc'] = 1
                            # 在白名单里面，货也够发
                    else:
                        # print s[0],' 属于白名单,但是fdc的库存够卖的'
                        # self.fdc_simu_orders_retail[f + d][order_index][s[0]] = self.orders_retail[f + d][order_index][s[0]]
                        tmp['fdc'] = 1
                        tmp['fdc_rdc'] = 1
                        if s[1] <= self.rdc_inv[index_rdc]:
                            tmp['rdc'] = 1
                    sku_state.append(copy.deepcopy(tmp))
                    # 标记订单类型,更新RDC库存，更新FDC库存
                flag_fdc = min([c['fdc'] for c in sku_state])
                flag_rdc = min([c['rdc'] for c in sku_state])
                flag_fdc_rdc = min([c['fdc_rdc'] for c in sku_state])
                if flag_fdc == 1:
                    self.orders_retail_type[o[1][0]] = 'fdc'
                    for s in dict(o[1][1]).items():
                        index = self.gene_index(f, s[0], d)
                        self.fdc_inv[index]['inv'] = self.fdc_inv[index]['inv'] - min(s[1], self.fdc_inv[index]['inv'])
                        self.fdc_simu_orders_retail[f + d][order_index][s[0]] = self.orders_retail[f + d][order_index][s[0]]
                        # self.logger.info('此订单从FDC发货，SKU的FDC库存');self.logger.info(index);self.logger.info(self.fdc_inv[index])
                elif flag_rdc == 1:
                    self.orders_retail_type[o[1][0]] = 'rdc'
                    for s in dict(o[1][1]).items():
                        index = self.gene_index('rdc', s[0], d)
                        self.simu_orders_retail[f + d][order_index][s[0]] = min(self.rdc_inv[index_rdc], s[1])
                        self.rdc_inv[index] = self.rdc_inv[index] - min(s[1], self.rdc_inv[index])
                        # self.logger.info('此订单从RDC发货，SKU的FDC库存');self.logger.info(index);self.logger.info(self.rdc_inv[index])
                elif flag_fdc_rdc == 1:
                    self.orders_retail_type[o[1][0]] = 'fdc_rdc'
                    for s in dict(o[1][1]).items():
                        index = self.gene_index(f, s[0], d)
                        rdc_index = self.gene_index('rdc', s[0], d)
                        #首先从fdc发货
                        self.fdc_simu_orders_retail[f + d][order_index][s[0]] = min(self.orders_retail[f + d][order_index][s[0]],self.fdc_inv[index]['inv'])
                        self.simu_orders_retail[f + d][order_index][s[0]] = min(self.orders_retail[f + d][order_index][s[0]],self.fdc_inv[index]['inv']+ self.rdc_inv[rdc_index])
                        # 首先从fdc 发货 其次不够的从rdc补，需求>=库存，则fdc库存为0,否则为 剩余量
                        sku_gap = s[1] - self.fdc_inv[index]['inv']
                        self.fdc_inv[index]['inv'] = 0 if sku_gap >= 0 else abs(sku_gap)
                        # 在模拟中有些订单会不被满足，所以需要在0 和实际值之间取最大值，无效订单在simu_order里面会被标记
                        self.rdc_inv[rdc_index] = self.rdc_inv[rdc_index] if sku_gap < 0 else max(
                            self.rdc_inv[rdc_index] - sku_gap, 0)
                else:
                    self.orders_retail_type[o[1][0]] = 'other'
                    for s in dict(o[1][1]).items():
                        index = self.gene_index(f, s[0], d)
                        rdc_index = self.gene_index('rdc', s[0], d)
                        # 首先从fdc发货
                        self.fdc_simu_orders_retail[f + d][order_index][s[0]] = min(self.orders_retail[f + d][order_index][
                            s[0]],self.fdc_inv[index]['inv'])
                        self.simu_orders_retail[f + d][order_index][s[0]] = min(self.orders_retail[f + d][order_index][
                            s[0]],self.fdc_inv[index]['inv']+ self.rdc_inv[rdc_index])
                        # 首先从fdc 发货 其次不够的从rdc补，需求>=库存，则fdc库存为0,否则为 剩余量
                        sku_gap = s[1] - self.fdc_inv[index]['inv']
                        self.fdc_inv[index]['inv'] = 0 if sku_gap >= 0 else abs(sku_gap)
                        # 在模拟中有些订单会不被满足，所以需要在0 和实际值之间取最大值，无效订单在simu_order里面会被标记
                        self.rdc_inv[rdc_index] = self.rdc_inv[rdc_index] if sku_gap < 0 else max(self.rdc_inv[rdc_index] - sku_gap, 0)
                        # 更新top-n字典,哪个fdc的订单被处理了，从哪个fdc抽取订单
                if all_rdc_orders_reatail.has_key(f):
                    if len(all_rdc_orders_reatail[f]) > 0:
                        top_n_min_orders_retail[f] = all_rdc_orders_reatail[f].pop(0)
                    else:
                        del all_rdc_orders_reatail[f]
            printruntime(t1, 'all_rdc_orders_reatail')
            # 更新下一天库存，将当天剩余库存标记为第二天库存,第二天到达库存会在开始增加上
            for f in self.fdc:
                for s in self.white_list[f]:
                    format_date = '%Y-%m-%d'
                    date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
                    date_s_c = date_tmp.strftime('%Y-%m-%d')
                    index_next = self.gene_index(f, s, date_s_c)
                    index = self.gene_index(f, s, d)
                    self.fdc_inv[index_next]['inv'] = self.fdc_inv[index]['inv']
            # 不仅仅更新白名单更新全量库存，如何获取全量list
            for s in self.all_sku_list:
                format_date = '%Y-%m-%d'
                date_tmp = datetime.datetime.strptime(d, format_date) + datetime.timedelta(1)
                date_s_c = date_tmp.strftime('%Y-%m-%d')
                index_next = self.gene_index('rdc', s, date_s_c)
                index = self.gene_index('rdc', s, d)
                self.rdc_inv[index_next] = self.rdc_inv[index]

            # 此处对 预测数据和明细数据进行更新self.fdc_max,self.fdc_safe,self.orders_retail
            # 删除已经使用的数据
            for f in self.fdc:
                for s in self.all_sku_list:
                    index_del = self.gene_index(f, s, d)
                    if self.fdc_safe.has_key(index_del):
                        self.fdc_safe[index_del]
                    if self.fdc_safe.has_key(index_del):
                        self.fdc_max[index_del]

            # 增加下一天的数据
            self.logger.info('update next day datas')
            start_date = datetime.datetime.strptime(d, '%Y-%m-%d') + datetime.timedelta(1)
            start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')
            if start_date not in self.date_range:
                continue
            tmp_sku_path = '/data0/cmo_ipc/inv_opt/Allocation_shell/datasets/data_total_actual/total_sku/' + start_date + '.pkl'
            pkl_sale = open(tmp_sku_path)
            sku_day_data = pickle.load(pkl_sale)
            pkl_sale.close()
            sku_day_data=sku_day_data[sku_day_data['white_flag']==1]
            # tmp_fdc_safe = pd.concat(
            #     [sku_day_data['date_s'].astype('str') + sku_day_data['dc_id'].astype('str')
            #      + sku_day_data['sku_id'].astype('str'),
            #      sku_day_data['forecast_daily_override_sales']], axis=1)
            # tmp_fdc_safe.columns = ['id', 'forecast_value']
            # tmp_fdc_safe = tmp_fdc_safe.set_index('id')['forecast_value'].to_dict()
            # self.fdc_safe.update(copy.deepcopy(tmp_fdc_safe))
            # tmp_fdc_max = pd.concat([sku_day_data['date_s'].astype('str') + sku_day_data['dc_id'].astype('str')
            #                                   + sku_day_data['sku_id'].astype('str'), sku_day_data['std']], axis=1)
            # tmp_fdc_max.columns = ['id', 'forecast_std']
            # tmp_fdc_max = tmp_fdc_max.set_index('id')['forecast_std'].to_dict()
            # self.fdc_max.update(copy.deepcopy(tmp_fdc_max))

            # 更白名单
            # white_list_dict=defaultdict(lambda :defaultdict(list))
            tmp_df = sku_day_data[sku_day_data['white_flag'] == 1][['date_s', 'sku_id', 'dc_id']]
            for k, v in tmp_df['sku_id'].groupby([tmp_df['date_s'], tmp_df['dc_id']]):
                self.white_list_dict[k[1]][k[0]] = list(v)
            self.logger.info('日sku数据更新完成')
            # 保存每天数据
            # if d in save_date:
            #     self.save_oneday(d)
            # 订单数据量不是太大，所以一次性全部加载到内存中

