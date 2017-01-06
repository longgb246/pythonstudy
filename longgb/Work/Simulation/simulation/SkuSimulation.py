# coding=utf-8

import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import rv_discrete
from matplotlib.backends.backend_pdf import PdfPages
from configServer import category_longTail_stable_DaysThreshold,category_longTail_stable_SalesThreshold,salespredictionErrorFilldays



class Utils:

    def __init__(self):
        pass

    @staticmethod
    def generate_date_range(start_dt, end_dt):
        if type(start_dt) == str:
            start_dt = datetime.datetime.strptime(start_dt, '%Y-%m-%d')
        if type(end_dt) == str:
            end_dt = datetime.datetime.strptime(end_dt, '%Y-%m-%d')
        step = datetime.timedelta(days=1)
        date_range = []
        while start_dt <= end_dt:
            date_range.append(start_dt.strftime('%Y-%m-%d'))
            start_dt += step
        return date_range

    @staticmethod
    def truncate_vlt_distribution(val, prob):
        vlt_mean = (val * prob).sum()
        val_trunc = np.copy(val[val >= vlt_mean])
        prob_trunc = np.copy(prob[val >= vlt_mean])
        prob_trunc[0] += np.sum(prob[~(val >= vlt_mean)])
        return val_trunc, prob_trunc

    @staticmethod
    def getCategory(sales,):
        category=99 #normal
        percent = (sum(sales>0)*1.0/ len(sales))
        salesMean = np.mean(sales)
        if  (percent>=category_longTail_stable_DaysThreshold) & (salesMean<=category_longTail_stable_SalesThreshold):
            category=1 #longTail_stable
        return category

    @staticmethod
    def getPredictionErrorMultiple(sales,pred_sales,cur_index):
        """
        judge whether prediction sales exceed the actual sales
        """
        sales3days =  sum([sales[cur_index]]*3)
        pred_sales3days = sum([pred_sales[cur_index][0]]*3)
        if cur_index >= 3:
            sales3days = sum(sales[cur_index-3:cur_index])
            pred_sales3days = sum(pred_sales[cur_index-3][0:3])
        multiple = max((sales3days*1.0/pred_sales3days),1)
        return multiple

    @staticmethod
    def getWeightedActSales(sales,cur_index):
        """
        1. estimate whether error is too large
        2. return weighted
        """
        if cur_index>= salespredictionErrorFilldays:
            actualSale = sales[cur_index-salespredictionErrorFilldays:cur_index]
            return [np.mean(actualSale)],[np.std(actualSale)]
        else:
            rang = salespredictionErrorFilldays - cur_index
            mean_sale = np.nanmean(sales[0:cur_index])
            actualSale = np.concatenate((sales[0:cur_index],np.array([mean_sale]*(rang))))
            return [np.mean(actualSale)],[np.std(actualSale)]

class SkuSimulation:

    def __init__(self, date_range, sales_his, inv_his, sales_pred_mean, sales_pred_sd, vlt_val, vlt_prob,
                 actual_pur_qtty, wh_qtn, pred_days=28, sku_id='', sku_name='', cr_pbs=None, bp_pbs=None,
                 lop_pbs=None, ti_pbs=None, org_nation_sale_num_band=None):
        self.sku_id = sku_id
        self.sku_name = sku_name
        self.date_range = date_range
        self.start_dt = datetime.datetime.strptime(self.date_range[0], '%Y-%m-%d')
        self.end_dt = datetime.datetime.strptime(self.date_range[1], '%Y-%m-%d')
        self.simulation_period = (self.end_dt - self.start_dt).days + 1

        self.sales_his_origin = sales_his
        self.inv_his = inv_his
        self.org_nation_sale_num_band = org_nation_sale_num_band

        # 销量填充
        self.sales_his = self.sales_his_origin.copy()
        stock_and_sale_days = ((self.inv_his > 0) & (self.sales_his > 0)).sum()
        stock_and_sale0_days = ((self.inv_his > 0) & (self.sales_his == 0)).sum()
        sales_his_mean = np.mean(self.sales_his[self.inv_his > 0])
        flag1 = (float(stock_and_sale0_days) / (stock_and_sale_days + 0.1)) > 0.5
        flag2 = sales_his_mean < 1
        # print flag1
        # print flag2
        if flag1 or flag2:
            # 零销量天数占比超过0.5或者平均销量小于1，使用随机抽样填充
            sales_set = self.sales_his_origin[self.inv_his > 0]
            np.random.seed(618)
            for i in np.where(self.inv_his == 0)[0]:
                self.sales_his[i] = np.random.choice(sales_set)
        else:
            self.sales_his[self.inv_his == 0] = np.median(self.sales_his[self.inv_his > 0])

        # 预测天数
        self.pred_days = pred_days
        # 销量预测（均值）
        self.sales_pred_mean = sales_pred_mean
        # 销量预测（标准差）
        self.sales_pred_sd = sales_pred_sd
        # 处理销量预测为空的情况：使用前一天的预测值
        for i in range(self.simulation_period):
            if self.sales_pred_mean[i] is None:
                self.sales_pred_mean[i] = self.sales_pred_mean[i - 1]
            if self.sales_pred_sd[i] is None:
                self.sales_pred_sd[i] = self.sales_pred_sd[i - 1]

        self.vlt_val = vlt_val.astype(np.int32)
        self.vlt_prob = vlt_prob
        self.vlt_distribution = rv_discrete(values=(self.vlt_val, self.vlt_prob))

        # 计算截断的VLT分布
        self.vlt_value_trunc, self.vlt_prob_trunc = Utils.truncate_vlt_distribution(self.vlt_val, self.vlt_prob)

        self.vlt_max = max(self.vlt_val)
        # 处理VLT最大值大于预测天数的情况：使用均值填充预测销量和预测标准差
        if self.vlt_max > self.pred_days:
            fill_days = self.vlt_max - self.pred_days
            for i in range(self.simulation_period):
                self.sales_pred_mean[i].extend([np.mean(self.sales_pred_mean[i])] * fill_days)
                self.sales_pred_sd[i].extend([np.mean(self.sales_pred_sd[i])] * fill_days)

        # 最晚的到货日期
        self.latest_arrive_index = 0

        # 将实际采购量换算成BP
        self.bp_his = []
        self.pur_cnt_his = 0
        self.actual_pur_qtty = actual_pur_qtty
        for i in range(self.simulation_period):
            if not np.isnan(self.actual_pur_qtty[i]/np.mean(self.sales_pred_mean[i])):
                if np.ceil(self.actual_pur_qtty[i]/np.mean(self.sales_pred_mean[i])) == np.inf:
                    self.bp_his.append(25)
                else:
                    self.bp_his.append(np.ceil(self.actual_pur_qtty[i]/np.mean(self.sales_pred_mean[i])))
                self.pur_cnt_his += 1

        self.wh_qtn = wh_qtn
        self.cr_pbs = cr_pbs
        self.bp_pbs = bp_pbs
        self.lop_pbs = lop_pbs
        self.ti_pbs = ti_pbs

        # 仿真库存，初始化
        # TODO：初始值=现货库存+在途
        self.inv_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.inv_sim[0] = self.inv_his[0]
        # 仿真销量
        self.sales_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        # 补货点
        self.lop = np.array([0] * self.simulation_period, dtype=np.float64)
        # 采购在途
        self.open_po_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        # 采购到货
        self.arrive_qtty_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        # 到货批次
        self.arrive_pur_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        # 仿真采购量
        self.pur_qtty_sim = [None] * self.simulation_period
        # 仿真VLT
        self.vlt_sim = [None] * self.simulation_period
        # 仿真采购次数
        self.pur_cnt_sim = 0
        # 采购成功次数
        self.success_cnt = 0
        # 初始化日期下标
        self.cur_index = 0
        # 仿真状态：{0: 未开始，1: 进行中，2：正常完成}
        self.simulation_status = 0
        self.category = Utils.getCategory(self.sales_his)
        self.subCategory = np.array([99]*self.simulation_period)

    def reset(self):
        self.inv_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.inv_sim[0] = self.inv_his[0]
        self.sales_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.lop = [0] * self.simulation_period
        self.open_po_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.arrive_qtty_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.arrive_pur_sim = np.array([0] * self.simulation_period, dtype=np.float64)
        self.pur_qtty_sim = [None] * self.simulation_period
        self.vlt_sim = [None] * self.simulation_period
        self.pur_cnt_sim = 0
        self.success_cnt = 0
        self.cur_index = 0
        self.simulation_status = 0
        self.latest_arrive_index = 0

    def calc_lop(self, cr=None):
        # 给定VLT，计算VLT期间总销量的均值
        demand_mean = [sum(self.sales_pred_mean[self.cur_index][:(l+1)]) for l in self.vlt_value_trunc]
        # VLT期间总销量均值的概率分布
        demand_mean_distribution = rv_discrete(values=(demand_mean, self.vlt_prob_trunc))
        part1 = demand_mean_distribution.mean()
        # 给定VLT，计算总销量的方差
        demand_var = [sum([i ** 2 for i in self.sales_pred_sd[self.cur_index][:(l+1)]]) for l in self.vlt_value_trunc]
        # demand_std = np.sqrt(demand_var)
        # VLT期间总销量方差的概率分布
        demand_var_distribution = rv_discrete(values=(demand_var, self.vlt_prob_trunc))
        # 条件期望的方差
        part21 = demand_mean_distribution.var()
        # 条件方差的期望
        part22 = demand_var_distribution.mean()
        # 计算补货点
        cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
        self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1))

    def calc_replenishment_quantity(self, bp=None, cr=None):
        cur_bp = self.bp_pbs[self.cur_index] if bp is None else bp
        sales_pred_sd = self.sales_pred_sd[self.cur_index]
        if cur_bp> len(sales_pred_sd):
            filldays = cur_bp - len(sales_pred_sd)
            sales_pred_sd = np.concatenate(( sales_pred_sd,[np.nanmean(sales_pred_sd)]*filldays))
        cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
        cur_bp_var = sum([sd**2 for sd in sales_pred_sd[:int(cur_bp)]])
        return np.ceil(self.lop[self.cur_index] +
                       cur_bp * np.mean(self.sales_pred_mean[self.cur_index]) +
                       norm.ppf(cur_cr) * math.sqrt(cur_bp_var) -
                       self.inv_sim[self.cur_index] -
                       self.open_po_sim[self.cur_index])

    def replenishment(self, bp=None):
        # 计算补货量
        self.pur_qtty_sim[self.cur_index] = self.calc_replenishment_quantity(bp)
        # 根据VLT分布生成到货时长
        self.vlt_sim[self.cur_index] = self.vlt_distribution.rvs()
        # 更新补货状态，last_index是采购到货天的下标
        last_index = self.cur_index + self.vlt_sim[self.cur_index] + 1

        # 判断上次的到货时间点早于该次的到货时间，如果晚于则重新抽样确定vlt
        while last_index < self.latest_arrive_index:
            self.vlt_sim[self.cur_index] = self.vlt_distribution.rvs()
            last_index = self.cur_index + self.vlt_sim[self.cur_index] + 1

        if last_index < self.simulation_period:
            # 更新采购在途
            self.open_po_sim[(self.cur_index+1):last_index] += self.pur_qtty_sim[self.cur_index]
            # 更新采购到货
            self.arrive_qtty_sim[last_index] += self.pur_qtty_sim[self.cur_index]
            # 更新到货批次
            self.arrive_pur_sim[last_index] += 1
            # 更新采购次数
            self.pur_cnt_sim += 1
        else:
            # 只更新采购在途
            self.open_po_sim[(self.cur_index+1):] += self.pur_qtty_sim[self.cur_index]
        # 更新next_arrive_index
        self.latest_arrive_index = last_index

    def run_simulation(self, cr=None, bp=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.simulation_status = 1
        while self.cur_index < self.simulation_period:
            # 如果有采购到货，将采购到货加到当天的现货库存中
            # 判断本次采购是否成功：当天0点库存大于0为成功
            if self.arrive_qtty_sim[self.cur_index] > 0:
                if self.inv_sim[self.cur_index] > 0:
                    self.success_cnt += self.arrive_pur_sim[self.cur_index]
                self.inv_sim[self.cur_index] += self.arrive_qtty_sim[self.cur_index]
            # 计算补货点
            self.calc_lop(cr)
            # 判断是否需要补货：现货库存 + 采购在途 < 补货点
            flag = (self.lop[self.cur_index] - self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index]) > 0
            if flag:
                self.replenishment(bp)
            # 当天仿真销量 = min(当天历史销量, 当天仿真库存量)
            self.sales_sim[self.cur_index] = min(self.sales_his[self.cur_index], self.inv_sim[self.cur_index])
            # 下一天的仿真库存量（初始库存） = 当天仿真库存量 - 当天销量
            if self.cur_index < self.simulation_period - 1:
                self.inv_sim[self.cur_index + 1] = self.inv_sim[self.cur_index] - self.sales_sim[self.cur_index]
            # 更新日期下标
            self.cur_index += 1
        self.simulation_status = 2

    def get_daily_data(self):
        if self.simulation_status == 2:
            daily_data = {'dt': Utils.generate_date_range(self.start_dt, self.end_dt),
                          'sales_his_origin': self.sales_his_origin,
                          'inv_his': self.inv_his,
                          'sales_sim': self.sales_sim,
                          'inv_sim': self.inv_sim,
                          'lop': self.lop,
                          'pur_qtty_sim': self.pur_qtty_sim,
                          'open_po_sim': self.open_po_sim,
                          'vlt_sim': self.vlt_sim,
                          'arrive_qtty_sim': self.arrive_qtty_sim,
                          'sales_pred_mean': self.sales_pred_mean,
                          'sales_pred_sd': self.sales_pred_sd,
                          'cr_pbs': self.cr_pbs,
                          'bp_pbs': self.bp_pbs,
                          'lop_pbs': self.lop_pbs,
                          'ti_pbs': self.ti_pbs,
                          'sku_id': np.repeat(self.sku_id, self.simulation_period),
                          'mean_price': np.repeat(self.wh_qtn, self.simulation_period),
                          'sale_num_band':self.org_nation_sale_num_band,
                          'category':self.category,
                          'subCategory' : self.subCategory,}
            return pd.DataFrame(daily_data, columns=['sku_id', 'dt', 'sales_his_origin', 'inv_his', 'sales_sim',
                                                     'mean_price', 'inv_sim', 'lop', 'pur_qtty_sim', 'open_po_sim',
                                                     'vlt_sim', 'arrive_qtty_sim', 'sales_pred_mean', 'sales_pred_sd',
                                                     'cr_pbs', 'bp_pbs', 'lop_pbs', 'ti_pbs','sale_num_band','category','subCategory'])
        else:
            print "Please call run_simulation() before getting daily data!"

    def calc_kpi(self):
        if self.simulation_status == 2:
            # 现货率（cr）：有货天数除以总天数
            cr_sim = (self.inv_sim > 0).sum() / float(self.simulation_period)
            cr_his = (self.inv_his > 0).sum() / float(self.simulation_period)
            # 周转天数（ito）：平均库存除以平均销量
            ito_sim = np.nanmean(self.inv_sim) / np.nanmean(self.sales_sim)
            ito_his = np.nanmean(self.inv_his) / np.nanmean(self.sales_his_origin)
            # 总销量（ts）
            ts_sim = np.sum(self.sales_sim)
            ts_his = np.sum(self.sales_his_origin)
            # gmv
            gmv_sim = ts_sim * self.wh_qtn
            gmv_his = ts_his * self.wh_qtn
            return [self.sku_id, cr_sim, cr_his, ito_sim, ito_his, gmv_sim, gmv_his, ts_sim, ts_his,
                    self.pur_cnt_sim, self.success_cnt, self.wh_qtn, self.pur_cnt_his,
                    self.org_nation_sale_num_band]
        else:
            print "Please call run_simulation() before calculating KPI!"

    def get_report(self, output_dir='./'):
        date_range = np.array(pd.to_datetime(Utils.generate_date_range(self.start_dt, self.end_dt)))
        inv0_date = date_range[self.inv_sim == 0]
        rep_date = date_range[~pd.isnull(self.pur_qtty_sim)]

        cr_sim = (self.inv_sim > 0).sum() / float(self.simulation_period)
        cr_his = (self.inv_his > 0).sum() / float(self.simulation_period)
        sales_pred_mean = map(np.mean, self.sales_pred_mean)
        sales_origin_mean = ((pd.Series(self.sales_his_origin)).rolling(window=28, center=False)).mean().shift(-27)
        max_inv = max(max(self.inv_his), max(self.inv_sim))
        max_sales = max(max(self.sales_his_origin), max(self.sales_sim), max(sales_pred_mean))

        # 库存-销量（历史）
        fig1 = plt.figure(figsize=(15, 15))
        axs1 = fig1.add_subplot(4, 1, 1)
        axs1.set_ylim(0, 1.1 * max_inv)
        axs1.plot(date_range, self.inv_his, label='inv_his', c='k')
        axs1.set_xlabel('date')
        axs1.set_ylabel('inv')
        axs1.legend(loc='upper left')

        axs2 = axs1.twinx()
        axs2.set_ylim(0, 1.1 * max_sales)
        axs2.plot(date_range, self.sales_his_origin, label='sales_his_origin', c='b')
        axs2.set_ylabel('sales')
        axs2.legend(loc='upper right')
        # 补货当天的竖线
        for date in rep_date:
            axs1.axvline(date, ls='-.', c='r')
        # 缺货天数的红线
        for date in inv0_date:
            axs1.plot([date, date + pd.Timedelta(days=1)], [0, 0], linewidth=5, c='r')
        # 添加title
        axs1.set_title('cr_sim:' + str(cr_sim) + '; cr_his:' + str(cr_his) + '; band:' + self.org_nation_sale_num_band)

        # 库存-销量（仿真）
        axs3 = fig1.add_subplot(4, 1, 2)
        axs3.set_ylim(0, 1.1 * max_inv)
        axs3.plot(date_range, self.inv_sim, label='inv_sim', c='k')
        axs3.set_xlabel('date')
        axs3.set_ylabel('inv')
        axs3.legend(loc='upper left')

        axs4 = axs3.twinx()
        axs4.set_ylim(0, 1.1 * max_sales)
        axs4.plot(date_range, self.sales_sim, label='sales_sim', c='b')
        axs4.plot(date_range, sales_pred_mean, label='sales_pred', c='g')
        axs4.set_ylabel('sales')
        axs4.legend(loc='upper right')

        # 补货当天的竖线
        for date in rep_date:
            axs3.axvline(date, ls='-.', c='r')
        # 缺货天数的红线
        for date in inv0_date:
            axs3.plot([date, date + pd.Timedelta(days=1)], [0, 0], linewidth=5, c='r')
            
        # 预测销量与真实销量对比图
        axs5 = fig1.add_subplot(4, 1, 3)
        axs5.plot(date_range, sales_pred_mean, c='r', label='sales_pred')
        axs5.plot(date_range, sales_origin_mean, c='b', label='sales_origin')
        axs5.set_xlabel('date')
        axs5.set_ylabel('sales')
        axs5.legend(loc='upper left')
        # 补货当天的竖线
        for date in rep_date:
            axs5.axvline(date, ls='-.', c='r')

        # 销量对比图
        axs6 = fig1.add_subplot(4, 1, 4)
        axs6.plot(date_range, self.sales_sim, c='r', label='sales_sim')
        axs6.plot(date_range, self.sales_his, c='b', label='sales_his')
        axs6.legend()

        # vlt分布图
        fig2 = plt.figure(figsize=(8, 4))
        axs6 = fig2.add_subplot(111)
        axs6.bar(self.vlt_val, self.vlt_prob, width=0.1)
        axs6.set_xlim(0, max(self.vlt_val) * 1.25)
        axs6.set_xlabel('vlt')
        axs6.set_ylabel('prob')
        axs6.set_title('vlt_distribution')

        figs = [fig1, fig2]

        pp = PdfPages(output_dir + str(self.sku_id) + '.pdf')

        for fig in figs:
            fig.savefig(pp, format='pdf')
            plt.close(fig)
        pp.close()
