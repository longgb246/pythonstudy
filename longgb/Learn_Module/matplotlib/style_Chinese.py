#-*- coding:utf-8 -*-
# 绿色：#6AB27B
# 土色：#a27712
# 浅紫色：#8172B2
# 蓝色：#4C72B0
# 红色：#C44E52
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.style as mstyle
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

print mstyle.available
plt.style.use('seaborn-darkgrid')
for each in mstyle.available:
    print each

