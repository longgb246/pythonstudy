# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


# ========================================================================
# =                                 功能函数                             =
# ========================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


from matplotlib.patches import ConnectionPatch
def plotEnlarge(data_x, data_y, scale=[], label=[], colors=[], linestyle=[], xlabel='X', ylabel='Y', title=['Origin Figure', 'Enlarge Figure']):
    '''
    data_x: list
    data_y: list
    scale: list  x的取值范围
    label: list  每条线对应的label
    colors: list  每条线对应的color
    linestyle：list  每条线对应的linestyle
    xlabel: str  x轴的label
    ylabel: str  y轴的label
    title: str  图的title
    '''
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(16, 7), dpi=98)  # 加个dpi调整。
    # ax1 = fig.add_subplot(121, aspect=5 / 2.5)
    ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122, aspect=5 / 2.5)
    ax2 = fig.add_subplot(122)
    if colors == []:
        colors = ['#6AB27B', '#C44E52', '#4C72B0', '#FFA455']*4
    if linestyle == []:
        linestyle = ['-', '--', '-.', ':']*4
    pair_data = []
    for i, x in enumerate(data_x):
        pair_data.append([pd.DataFrame(np.array([x, data_y[i]]).T, columns=['x', 'y'])])
        label_tmp = 'line_{0}'.format(i + 1) if label == [] else label[i]
        ax1.plot(x, data_y[i], color=colors[i], linestyle=linestyle[i], label=label_tmp, linewidth=2)
        ax2.plot(x, data_y[i], color=colors[i], linestyle=linestyle[i], label=label_tmp, linewidth=2)
    x_lim = ax1.get_xlim()
    x_range = x_lim[1] - x_lim[0]
    # ax1.axis([0.0, 5.01, -1.0, 1.5])
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_title(title[0],fontsize=18)
    ax1.grid(True)
    ax1.legend(loc='best')
    # ax1.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")
    if scale == []:
        tx0 = x_lim[1] - x_range * 0.2
        tx1 = x_lim[1] - x_range * 0.1
        ty0 = np.nan
        ty1 = np.nan
    else:
        tx0 = scale[0]
        tx1 = scale[1]
        ty0 = np.nan if len(scale) < 3 else scale[2]
        ty1 = np.nan if len(scale) < 4 else scale[3]
    for i, each in enumerate(pair_data):
        each = each[0]
        tmp_max = np.max(each[(each['x'] >= tx0) & (each['x'] <= tx1)]['y'])
        tmp_min = np.min(each[(each['x'] >= tx0) & (each['x'] <= tx1)]['y'])
        if i == 0:
            y_max = tmp_max
            y_min = tmp_min
        else:
            y_max = y_max if y_max > tmp_max else tmp_max
            y_min = y_min if y_min < tmp_min else tmp_min
    y_range = y_max - y_min
    ty0 = ty0 if ty0 != np.nan else y_min - y_range * 0.16
    ty1 = ty1 if ty1 != np.nan else y_max + y_range * 0.16
    ax2.set_xlim(tx0, tx1)
    ax2.set_ylim(ty0, ty1)
    # ax2.axis([tx0, tx1, ty0, ty1])          # 设置不同的范围
    # ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_title(title[1], fontsize=18)
    ax2.grid(True)
    ax2.legend(loc='best')
    sx = [tx0, tx1, tx1, tx0, tx0]          # 画方框
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax1.plot(sx, sy, "purple")
    # plot patch lines
    y_a = 0.05 * (ty1 - ty0)
    x_a = 0.05 * (tx1 - tx0)
    xy = (tx1 - x_a, ty1 - y_a)
    xy2 = (tx0 + x_a, ty1 - y_a)
    # 重点：连接线
    con = ConnectionPatch(xyA=xy2, xyB=xy,
                          coordsA="data", coordsB="data",           # 这个参数必须要！
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)                      # 在p2上添加
    xy = (tx1 - x_a, ty0 + y_a)
    xy2 = (tx0 + x_a, ty0 + y_a)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)
    plt.show()
    return [fig, ax1, ax2]


# ========================================================================
# =                                 实用函数                             =
# ========================================================================
def script_01_plotSigmoid():
    data_x = [np.arange(-60,60,1)]
    data_y = [sigmoid(data_x[0])]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x, y, color='#4C72B0', linewidth=2)
    plotEnlarge(data_x, data_y, scale=[-6, 6, 0, 1])


if __name__ == '__main__':
    pass
