#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
参赛者提交代码示例
sample submission
'''

# import all modules been used 
import pandas as pd
import numpy as np

class UserPolicy:
    def __init__(self, initial_inventory, inventory_replenishment, sku_demand_distribution, sku_cost  ):
        self.inv = [initial_inventory]
        self.replenish = inventory_replenishment
        self.distribution = sku_demand_distribution
        self.cost = sku_cost
        self.sku_limit = np.asarray([200, 200, 200, 200, 200])
        self.extra_shipping_cost_per_unit = 0.01
        self.capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])

    def daily_decision(self,t):
        '''
        daily decision of inventory allocation
        input values:
            t: decision date
        return values:
            allocation decision, 2-D numpy array, shape (5,1000), type integer
        '''
        # Your algorithms here
        # simple rule: no transshipment at all
        transshipment_decision = np.zeros((5, 1000)).astype(int)

        return transshipment_decision

    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)

    def some_other_functions():
        pass

    
