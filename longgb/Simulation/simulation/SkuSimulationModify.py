# coding=utf-8
import os
from sys import path
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
print pth
import numpy as np
from com.jd.pbs.simulation.SkuSimulation import SkuSimulation
from com.jd.pbs.utils.MixtureGaussian import MixtureGaussian
import math
from scipy.stats import rv_discrete,norm
from com.jd.pbs.simulation.SkuSimulation import Utils
from com.jd.pbs.simulation.configServer import Sales_prediction_errorPercent
class SkuSimulationBp25(SkuSimulation):

    def calc_replenishment_quantity(self, bp=None):
        if bp is None:
            raise Exception("Default bp must be set!")
        # 如果没有采购记录，取默认BP，否则，取采购历史BP的均值
        cur_bp = bp if self.pur_cnt_his == 0 else np.mean(self.bp_his)
        # 如果BP大于25，截断
        cur_bp = cur_bp if cur_bp <= 25 else 25
        return np.ceil(self.lop[self.cur_index] + cur_bp * np.mean(self.sales_pred_mean[self.cur_index]) -
                       self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index])


class SkuSimulationMg(SkuSimulation):

    def calc_lop(self, cr=None):
        demand_mean = [sum(self.sales_pred_mean[self.cur_index][:(l+1)]) for l in self.vlt_val]
        demand_var = [sum([i ** 2 for i in self.sales_pred_sd[self.cur_index][:(l+1)]]) for l in self.vlt_val]
        demand_std = np.sqrt(demand_var)
        mg = MixtureGaussian(demand_mean, demand_std, self.vlt_prob)
        cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
        self.lop[self.cur_index] = np.ceil(mg.inverse_cdf(cur_cr))


class SkuSimulationPbs(SkuSimulation):

    def calc_lop(self, cr=None):
        self.lop[self.cur_index] = self.lop_pbs[self.cur_index]

    def calc_replenishment_quantity(self, bp=None):
        return self.ti_pbs[self.cur_index] - self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index]

    def run_simulation(self, cr=None, bp=None,seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.simulation_status = 1
        while self.cur_index < self.simulation_period:
            # 如果有采购到货，将采购到货加到当天的现货库存中
            # 判断本次采购是否成功：当天0点库存大于0为成功
            if self.arrive_qtty_sim[self.cur_index] > 0:
                if self.inv_sim[self.cur_index] > 0:
                    self.success_cnt += 1
                self.inv_sim[self.cur_index] += self.arrive_qtty_sim[self.cur_index]
            # 计算补货点
            self.calc_lop(cr)
            # 判断是否需要补货：现货库存 + 采购在途 < 补货点
            flag1 = (self.lop[self.cur_index] - self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index]) >= 0
            # 如果（目标库存-现货库存-在途库存）小于0，不补货
            flag2 = (self.ti_pbs[self.cur_index] - self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index]) >= 0
            if flag1 and flag2:
                self.replenishment(bp)
                self.pur_cnt_sim += 1
            # 当天仿真销量 = min(当天历史销量, 当天仿真库存量)
            self.sales_sim[self.cur_index] = min(self.sales_his[self.cur_index], self.inv_sim[self.cur_index])
            # 下一天的仿真库存量（初始库存） = 当天仿真库存量 - 当天销量
            if self.cur_index < self.simulation_period - 1:
                self.inv_sim[self.cur_index + 1] = self.inv_sim[self.cur_index] - self.sales_sim[self.cur_index]
            # 更新日期下标
            self.cur_index += 1
        self.simulation_status = 2

class MaxVlt_Times_Demand(SkuSimulation):
    def calc_lop(self, cr=None,type="prob"):
        if self.org_nation_sale_num_band in ['A','B']:
            vlt_max= int(self.vlt_max)
            fill_days = int(vlt_max-self.pred_days)
            sales_pred_mean = self.sales_pred_mean[self.cur_index]
            sales_pred_sd = self.sales_pred_sd[self.cur_index]
            if fill_days>0:
                sales_pred_mean.extend([np.nanmean(sales_pred_mean)]*fill_days)
                sales_pred_sd.extend([np.nanmean(sales_pred_sd)]*fill_days)
            # 给定VLT，计算VLT期间总销量的均值
            demand_mean = [sum(sales_pred_mean[:(vlt_max+1)])]
            vlt_prob = 1
            # VLT期间总销量均值的概率分布
            demand_mean_distribution = rv_discrete(values=(demand_mean, vlt_prob))
            part1 = demand_mean_distribution.mean()
            # 给定VLT，计算总销量的方差
            demand_var = [sum([i ** 2 for i in sales_pred_sd[:int(vlt_max+1)]])]
            # demand_std = np.sqrt(demand_var)
            # VLT期间总销量方差的概率分布
            demand_var_distribution = rv_discrete(values=(demand_var, vlt_prob))
            # 条件期望的方差
            part21 = demand_mean_distribution.var()
            # 条件方差的期望
            part22 = demand_var_distribution.mean()
            # 计算补货点
            cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
            self.subCategory[self.cur_index] = 66
            self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1))
        else:
            SkuSimulation.calc_lop(self,cr)

class SkuSimulationSalesCorrection(SkuSimulation):
    def calc_lop(self, cr=None):
        if self.category == 99:
            SkuSimulation.calc_lop(self,cr)
        elif self.category == 1:
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
            self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * 2*math.sqrt(part21 + part22 + 0.1))


class HisSkuBpMeanSimulation(SkuSimulation):
    def calc_replenishment_quantity(self, bp=None,cr=None):
        cur_bp = np.mean(self.bp_his)
        if cur_bp:
            cur_bp = 25
        return np.ceil(self.lop[self.cur_index] + cur_bp* np.mean(self.sales_pred_mean[self.cur_index]) -
        self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index])

class LongTailLowSalesSimulation(SkuSimulation):
    def calc_lop(self, cr=None):
        if self.category == 99: #其他
            SkuSimulation.calc_lop(self,cr)
        elif self.category == 1: #长尾低销量
            # 给定VLT，计算VLT期间总销量的均值
            multiple = Utils.getPredictionErrorMultiple(self.sales_his,self.sales_pred_mean,self.cur_index)
            print "---------------multiple---------------" ,multiple ,self.sku_id
            if multiple >= Sales_prediction_errorPercent:
                sales_pred_mean,sales_pred_sd = Utils.getWeightedActSales(self.sales_his,self.cur_index)
                fill_days = max(self.vlt_value_trunc)
                sales_pred_mean = np.array(sales_pred_mean * fill_days)
                sales_pred_sd = np.array(sales_pred_sd * fill_days)
                print sales_pred_mean
                print sales_pred_sd
                demand_mean = [sum(sales_pred_mean[:(l+1)]) for l in self.vlt_value_trunc]
                # VLT期间总销量均值的概率分布
                demand_mean_distribution = rv_discrete(values=(demand_mean, self.vlt_prob_trunc))
                part1 = demand_mean_distribution.mean()
                # 给定VLT，计算总销量的方差
                demand_var = [sum([i ** 2 for i in sales_pred_sd[:(l+1)]]) for l in self.vlt_value_trunc]
                # demand_std = np.sqrt(demand_var)
                # VLT期间总销量方差的概率分布
                demand_var_distribution = rv_discrete(values=(demand_var, self.vlt_prob_trunc))
                # 条件期望的方差
                part21 = demand_mean_distribution.var()
                # 条件方差的期望
                part22 = demand_var_distribution.mean()
                # 计算补货点
                cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
                self.subCategory[self.cur_index] = 66
                self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1))
            else:
                SkuSimulation.calc_lop(self,cr)

class PreSalesMonitor(SkuSimulation):
    def calc_lop(self, cr=None):
        # 给定VLT，计算VLT期间总销量的均值
        multiple = Utils.getPredictionErrorMultiple(self.sales_his,self.sales_pred_mean,self.cur_index)
        if multiple >= Sales_prediction_errorPercent:
            sales_pred_mean,sales_pred_sd = Utils.getWeightedActSales(self.sales_his,self.cur_index)
            fill_days = max(self.vlt_value_trunc)
            sales_pred_mean = np.array(sales_pred_mean * fill_days)
            sales_pred_sd = np.array(sales_pred_sd * fill_days)
            demand_mean = [sum(sales_pred_mean[:(l+1)]) for l in self.vlt_value_trunc]
            # VLT期间总销量均值的概率分布
            demand_mean_distribution = rv_discrete(values=(demand_mean, self.vlt_prob_trunc))
            part1 = demand_mean_distribution.mean()
            # 给定VLT，计算总销量的方差
            demand_var = [sum([i ** 2 for i in sales_pred_sd[:(l+1)]]) for l in self.vlt_value_trunc]
            # demand_std = np.sqrt(demand_var)
            # VLT期间总销量方差的概率分布
            demand_var_distribution = rv_discrete(values=(demand_var, self.vlt_prob_trunc))
            # 条件期望的方差
            part21 = demand_mean_distribution.var()
            # 条件方差的期望
            part22 = demand_var_distribution.mean()
            # 计算补货点
            cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
            self.subCategory[self.cur_index] = 66
            self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1))
        else:
            SkuSimulation.calc_lop(self,cr)
