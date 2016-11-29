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
        vlt_max= 1.5 * self.vlt_max
        sales_pred_mean = self.sales_pred_mean.extend(self.sales_pred_mean)
        sales_pred_sd = self.sales_pred_sd.extend(self.sales_pred_sd)
        self.lop[self.cur_index]= sales_pred_mean[self.cur_index][:vlt_max].sum()

        # 给定VLT，计算VLT期间总销量的均值
        demand_mean = [sum(sales_pred_mean[self.cur_index][:vlt_max+1])]
        vlt_prob = 1
        # VLT期间总销量均值的概率分布
        demand_mean_distribution = rv_discrete(values=(demand_mean, vlt_prob))
        part1 = demand_mean_distribution.mean()
        # 给定VLT，计算总销量的方差
        demand_var = [sum([i ** 2 for i in sales_pred_sd[self.cur_index][:vlt_max+1]])]
        # demand_std = np.sqrt(demand_var)
        # VLT期间总销量方差的概率分布
        demand_var_distribution = rv_discrete(values=(demand_var, vlt_prob))
        # 条件期望的方差
        part21 = demand_mean_distribution.var()
        # 条件方差的期望
        part22 = demand_var_distribution.mean()
        # 计算补货点
        cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
        self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1))

class SkuSimulationSalesCorrection(SkuSimulation):
    # 如果销量预测出现较大偏差，使用历史28天平均销量代替
    # def calc_replenishment_quantity(self, bp=None):
    #     cur_bp = self.bp_pbs[self.cur_index] if bp is None else bp
        # #处理预测销量为0的情况
        # if self.cur_index<28:
        #     sales_28days=self.sales_sim[0:self.cur_index+1]
        #     pred_28days=sum(self.sales_pred_mean[0][0:self.cur_index])
        # else:
        #     sales_28days=self.sales_sim[self.cur_index-27:self.cur_index+1]
        #     pred_28days=sum(self.sales_pred_mean[self.cur_index-28])
        #     #获取过去28天的销量
        # total_sales_28days=sum(sales_28days)
        # if total_sales_28days==0:
        #     #如果库存和在途为0，则补一个，否则不补货
        #     return max(1 -self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index],0)
        # if sum(self.sales_pred_mean[self.cur_index])==0:
        #     #如果只有一天有销量，则目标库存为max(2,销量)，所以补货量为目标库存-在途-库存
        #     if np.count_nonzero(sales_28days)==1:
        #         return max(min(2,max(sales_28days))-self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index],0)
        #     #如果有一天以上的销量补货量为28天的销量和
        #     else:
        #         return max(np.sum(sales_28days[sales_28days>0])-self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index],0)
        # #预测销量与实际销量偏差较大,如果cur_index<28天如何判断,直接获取对应天数的预测与实际值
        # elif (abs(float(pred_28days-total_sales_28days))/total_sales_28days)>0.3 and float(pred_28days-total_sales_28days)>5:
        #     return max(total_sales_28days-self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index],0)
        # #其他情况
        # else:
        #     return np.ceil(self.lop[self.cur_index] + cur_bp * np.mean(self.sales_pred_mean[self.cur_index]) -
        #                self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index])
        # return np.ceil(self.lop[self.cur_index] + cur_bp * np.mean(self.sales_pred_mean[self.cur_index]) -
        #                self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index])
    def calc_lop(self, cr=None):
        # if self.cur_index<28:
        #     sales_28days=self.sales_sim[0:self.cur_index+1]
        #     pred_28days=sum(self.sales_pred_mean[0][0:self.cur_index])
        # else:
        #     sales_28days=self.sales_sim[self.cur_index-27:self.cur_index+1]
        #     pred_28days=sum(self.sales_pred_mean[self.cur_index-28])
        # total_sales_28days=sum(sales_28days)
        # #如果过去28天销量为0，将补货点置为1
        # if total_sales_28days==0:
        #     self.lop[self.cur_index]=1
        # elif sum(self.sales_pred_mean[self.cur_index])==0:
        #     #如果只有一天有销量，则补货点为1
        #     if np.count_nonzero(sales_28days)==1:
        #         self.lop[self.cur_index]=1
        #     #如果有一天以上的销量补货量为28天的销量和的0.2倍向上取整
        #     else:
        #         self.lop[self.cur_index]=np.ceil(total_sales_28days*0.2)
        # elif (abs(float(pred_28days-total_sales_28days))/total_sales_28days)>0.3 and float(pred_28days-total_sales_28days)>5:
        #     self.lop[self.cur_index]=np.ceil(total_sales_28days*0.2)
        # else:
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
        self.lop[self.cur_index] = np.ceil(part1 + norm.ppf(cur_cr) * 2*math.sqrt(part21 + part22 + 0.1))


class HisSkuBpMeanSimulation(SkuSimulation):
    def calc_replenishment_quantity(self, bp=None,cr=None):
        cur_bp = np.mean(self.bp_his)
        if cur_bp:
            cur_bp = 25
        return np.ceil(self.lop[self.cur_index] + cur_bp* np.mean(self.sales_pred_mean[self.cur_index]) -
        self.inv_sim[self.cur_index] - self.open_po_sim[self.cur_index])

