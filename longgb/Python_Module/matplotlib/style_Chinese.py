#-*- coding:utf-8 -*-
# 绿色：#6AB27B
# 土色：#a27712
# 浅紫色：#8172B2
# 蓝色：#4C72B0
# 红色：#C44E52
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def useChinese():
    import matplotlib.style as mstyle
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

def useStyle():
    import matplotlib.style as mstyle
    print mstyle.available
    plt.style.use('seaborn-darkgrid')
    for each in mstyle.available:
        print each

def useSelfStyle():
    import matplotlib.pyplot as plt
    plt.style.use('mystyle')
    a = np.random.rand(100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(a)
    ax.set_ylabel(u'画     \ny     \n轴     ', rotation=0, size=15)
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2, alpha=0.6)
    bbox2_props = dict(boxstyle="round,pad=0.3", fc="#797979", ec="black", lw=2, alpha=0.6)
    t = ax.text(50, 0.45, "Direction", ha="center", va="center", rotation=45, size=15, bbox=bbox_props, alpha=0.6)
    t = ax.text(40, 0.3, "TEST1", ha="center", va="center", size=20, bbox=bbox2_props, alpha=0.6)
    t = ax.text(60, 0.6, "TEST2", ha="center", va="center", size=20, bbox=bbox2_props, alpha=0.6)
    plt.tight_layout()


if __name__ == '__main__':
    pass


