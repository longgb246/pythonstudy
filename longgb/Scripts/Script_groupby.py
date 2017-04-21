# -*- coding:utf-8 -*-
# ========================== 测试groupby ==========================
import pandas as pd
import numpy as np

a = pd.DataFrame(np.matrix([[1, 2], [1, 1], [2, 3], [2, 11]]), columns=['a', 'b'])

for sku, group in a.groupby(['a']):
    print sku
    print group
    print group.reset_index()

# ========================== 测试排序 ==========================
a = np.array([23, 1, 24, 12, 412, 4, 23, 41, 234, 1243, 323, 251, 1243, 1])
b = np.argsort(a)
c = np.sort(a)
# [23, 1, 24, 12, 412, 4, 23, 41, 234, 1243, 323, 251, 1243, 1]
# [ 1, 13,  5,  3,  0,  6,  2,  7,  8, 11, 10,  4,  9, 12]


mm = pd.DataFrame([a, b, c]).T
mm.columns = list('abc')
mm.sort_values(['a', 'b'], ascending=[1, 0])


import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2],[2,2])
