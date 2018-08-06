#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""
simulation code for submission
usage: python3 simulation.py
"""
import pandas as pd
import numpy as np
import time
import traceback


'''仿真逻辑'''
class DailyEvaluate:
    def __init__(self,
        sku_list, dc_list, days,
        initial_inventory, inventory_replenishment,
        sku_limit, capacity_limit,
        abandon_rate,  sku_cost, extra_shipping_cost_per_unit,   
        true_demand):

        self.sku_list = sku_list
        self.dc_list = dc_list
        self.days = days

        '''transform initial_inventory into numpy array, shape (6,1000) '''
        df_inv = initial_inventory.sort_values(['dc_id','item_sku_id'])
        self.inv = np.asarray([df_inv.loc[df_inv.dc_id == dcid].stock_quantity.values for dcid in dc_list])

        '''transform inventory_replenishment into numpy array, shape (29, 1000) '''
        df_rep = pd.DataFrame([[d,skuid] for d in range(1, self.days+1) for skuid in  self.sku_list], columns = ['date','item_sku_id'])
        df_rep = pd.merge(df_rep, inventory_replenishment[['item_sku_id', 'date', 'replenish_quantity']], on =['date','item_sku_id'], how = 'left')
        df_rep.fillna(value = 0, inplace = True)
        self.inventory_replenishment =  np.asarray([df_rep.loc[df_rep.date == d].replenish_quantity.values for d in range(1,self.days+1)], dtype= int)

        self.sku_limit  = sku_limit
        self.capacity_limit = capacity_limit
        self.abandon_rate = abandon_rate

        df_cost = sku_cost.sort_values('item_sku_id', ascending = True)
        self.sku_cost = df_cost.stockout_cost.values

        self.extra_shipping_cost_per_unit = extra_shipping_cost_per_unit

        self.true_demand = true_demand

        self.t = 1
        self.shortage_cost = 0
        self.extra_shipping_cost =0


    def daily_update(self, new_transshipment, t):
        # constraint check
        try:
            # decision format: shape (number of fdc, number of sku)
            assert new_transshipment.shape == (len(self.dc_list)-1, len(self.sku_list)), 'invalid decision format '
            # inventory decision should be nonnegative integers
            assert np.all(new_transshipment >= 0), 'negative transshipment'
            assert np.all(np.ceil(new_transshipment) == np.floor(new_transshipment)), 'integrity constraint violation'
            new_transshipment = new_transshipment.astype(int)

            # 调拨sku种类限制
            # constraint on transship sku variety for each fdc
            assert np.all(np.count_nonzero(new_transshipment, axis = 1) <= self.sku_limit), 'sku limits violation'

            #调拨件数限制
            # constraint on transship capacity for each fdc
            assert np.all(new_transshipment.sum(axis = 1) <= self.capacity_limit), 'capacity limits violation'

            # 1 根据调拨决策, 更新可售与在途库存
            # update rdc on hand inventory, fdc intransit transship inventory according to new_transshipment
            self.inv[0] -= new_transshipment.sum(axis =0)
            # 调拨后RDC库存非负
            # the rdc inventory should be nonnegative after transshipment
            assert np.all(self.inv[0] >= 0), 'transshipment should not greater than RDC inventory'
            self.intransit_transshipment = new_transshipment

        except Exception as e:
            # invalid decision
            # print(e)
            return {'status':False, 'constraints_violation':e}

        # 2. 当天需求实现，更新当天剩余库存，计算当天缺货、持货、rdc直发订单的额外运输成本 
        # demand realization, record leftover inventory and corresponding costs    
        # fdc local inventory only fulfills local demand
        _fdc_sales = np.minimum(self.true_demand[t][1:], self.inv[1:])
        _spill_over = np.floor(((self.true_demand[t][1:]-_fdc_sales).T * (1-self.abandon_rate)).T)
        # spillover demand is firstly fulfilled by intransit transshipment to that FDC
        _transship_sales = np.minimum(_spill_over, self.intransit_transshipment) 
        _spill_over -= _transship_sales

        # rdc inventory fulfills rdc demand and fdc spillover demand 
        _rdc_tot_demand =  self.true_demand[t][0] + _spill_over.sum(axis=0)
        _rdc_sales =  np.minimum(_rdc_tot_demand, self.inv[0])

        # deduct inventory has been sold 
        self.inv[1:] -= _fdc_sales.astype(int)
        self.intransit_transshipment -= _transship_sales.astype(int)
        self.inv[0] -= _rdc_sales.astype(int)

        # calculate inventory cost
        _lost_sale = self.true_demand[t].sum(axis=0) - (_fdc_sales.sum(axis=0) + _transship_sales.sum(axis = 0) + _rdc_sales)
        self.shortage_cost += (_lost_sale* self.sku_cost).sum()
        self.extra_shipping_cost += self.extra_shipping_cost_per_unit * np.maximum(_rdc_sales - self.true_demand[t][0], np.zeros(len(self.sku_list)).astype(int) ).sum()

        # 3. 接收到达的在途库存（补货+调拨） 
        # update on hand inventory after receiving intransit replenishment and transshipment
        self.inv[0] += self.inventory_replenishment[0]
        self.inventory_replenishment = self.inventory_replenishment[1:]
        self.inv[1:] += self.intransit_transshipment
        return {'status':True}


if __name__ == '__main__': 
    timestart = time.time()

    # 参赛者提交的脚本文件名
    # contestant's submission
    from submission import UserPolicy

    # 参数设置
    np.random.seed(1)

    #cost parameters
    extra_shipping_cost_per_unit = 0.01
    sku_cost = pd.read_csv('sku_cost.csv')

    # sku_list demand information, mean var quantiles
    sku_demand_distribution = pd.read_csv('sku_demand_distribution.csv') 

    # initial on hand inventory
    initial_inventory = pd.read_csv('initial_inventory.csv')

    # intransit replenishment
    inventory_replenishment = pd.read_csv('inventory_replenishment.csv') 

    # maximum daily number of unique products that can be allocated to FDC 
    sku_limit = np.asarray([200, 200, 200, 200, 200])

    # maximum daily total number of units that can be allocated to FDC
    capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])

    # simulation period length
    days = 30

    # DC Number, RDC id: 0, FDC id: 1~5
    dc_list  = list(range(6))

    # sku types
    sku_list  = [i for i in range(1, 1001)]

    #因本地缺货，时效变慢而放弃购买的比例
    # ratio of customers that abandon purchase due to local stock out
    abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])

    # 100次仿真所用到的真实销量数据
    # demand data for 100 simulation runs, generated from some distributions
    number_of_sample = 100
    # FAKE DEMAND, randomly generated
    demand_sample_all = np.random.randint(10, size = (number_of_sample, days, len(dc_list), len(sku_list)))
    


    score = 0
    '''100 simulation runs'''
    for sample in range(number_of_sample):
        print('sample ',sample)

        # run simulation for each demand sample
        '''instance of simulation'''
        instance = DailyEvaluate(sku_list, dc_list, days, 
            initial_inventory, inventory_replenishment, 
            sku_limit, capacity_limit, 
            abandon_rate, sku_cost, extra_shipping_cost_per_unit, 
            demand_sample_all[sample])

        '''instance of inventory policy'''
        some_policy = UserPolicy(initial_inventory.copy(deep=True), inventory_replenishment.copy(deep=True),\
             sku_demand_distribution.copy(deep=True), sku_cost.copy(deep=True))

        for t in range(1, days+1):
            transship_decision = some_policy.daily_decision(t)
            update_return = instance.daily_update(transship_decision, t-1)
            if update_return['status']:
                if t <days:
                    tday_inventory = pd.DataFrame([[s, d, instance.inv[d][s-1] ] for d in dc_list for s in sku_list ], \
                                     columns = ['item_sku_id','dc_id','stock_quantity'])            
                    some_policy.info_update(tday_inventory,t)
            else:
                print('constraint violation in period ', t, '\n', update_return['constraints_violation']) 
                break
        if update_return['status']:
            score += instance.shortage_cost + instance.extra_shipping_cost
            # print(instance.shortage_cost + instance.extra_shipping_cost)
        else:
            break

    '''
    最终分数，100次仿真的成本均值，越低越好
    Final score of the policy, lower is better
    '''
    score = score/100
    timeend = time.time() - timestart
    print('time', timeend)
    print('score',score)

