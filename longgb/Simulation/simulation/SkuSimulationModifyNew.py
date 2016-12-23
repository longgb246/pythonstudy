# coding=utf-8

import math
import numpy as np
from com.jd.pbs.simulation.SkuSimulationNew import SkuSimulation
from scipy.stats import rv_discrete
from scipy.stats import norm


class SkuSimulationPredCorrection(SkuSimulation):

    def calc_lop(self, cr=None):
        his_period = 14
        if (self.cur_index - his_period) >= 0:
            # 计算历史天的平均销量
            avg_sales_his = np.mean(self.sales_his_origin[(self.cur_index-his_period):self.cur_index])
            # 计算未来28天预测平均销量
            avg_sales_pred28 = np.mean(self.sales_pred_mean[self.cur_index])
            # 偏差过大，使用历史销量计算补货点
            if (avg_sales_his / avg_sales_pred28) > 1.5:
                print True
                sd_sales_his14 = np.std(self.sales_his_origin[(self.cur_index-his_period):self.cur_index])
                # 给定VLT，计算VLT期间总销量的均值
                demand_mean = [l * avg_sales_his for l in self.vlt_value_modify]
                # VLT期间总销量均值的概率分布
                demand_mean_distribution = rv_discrete(values=(demand_mean, self.vlt_prob_modify))
                part1 = demand_mean_distribution.mean()
                # 给定VLT，计算总销量的方差
                demand_var = [l * sd_sales_his14 ** 2 for l in self.vlt_value_modify]
                # VLT期间总销量方差的概率分布
                demand_var_distribution = rv_discrete(values=(demand_var, self.vlt_prob_modify))
                # 条件期望的方差
                part21 = demand_mean_distribution.var()
                # 条件方差的期望
                part22 = demand_var_distribution.mean()
                # 计算补货点
                cur_cr = self.cr_pbs[self.cur_index] if cr is None else cr
                self.lop[self.cur_index] = (np.ceil(part1 + norm.ppf(cur_cr) * math.sqrt(part21 + part22 + 0.1)))
            else:
                SkuSimulation.calc_lop(self, cr=cr)
        else:
            SkuSimulation.calc_lop(self, cr=cr)
