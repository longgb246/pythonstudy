#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

a = pd.DataFrame(np.matrix([[1,2],[1,1],[2,3],[2,11]]), columns=['a','b'])

for sku, group in a.groupby(['a']):
    print sku
    print group
    print group.reset_index()



