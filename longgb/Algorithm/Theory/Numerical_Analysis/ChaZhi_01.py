#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 自定义排序
# a = [[3,1],[2,3],[124,23,12]]
# a.sort(key=lambda x: x[1], reverse=True)              # 降序
# b = sorted(a, key=lambda x: x[1], reverse=False)      # 建议使用这个


def ChaZhi_Linear(points_2, plot=True):
    '''
    线性插值
    points_2  likes  [[3,5], [6,2]]
    points_2  = [[3,5], [6,2]]
    '''
    points_2 = sorted(points_2, key=lambda x: x[0])
    x_1 = points_2[0]
    x_2 = points_2[1]
    x_range = np.linspace(x_1[0], x_2[0], 100)
    y = np.array([(x - x_2[0])/(x_1[0] - x_2[0])*x_1[1] + (x - x_1[0])/(x_2[0] - x_1[0])*x_2[1] for x in x_range])
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_range, y)
        x_lim_min, x_lim_max = plt.xlim()
        ax.set_xlim(x_lim_min - (x_lim_max - x_lim_min) * 0.1, x_lim_max + (x_lim_max - x_lim_min) * 0.1)
        y_lim_min, y_lim_max = plt.ylim()
        ax.set_ylim(y_lim_min - (y_lim_max - y_lim_min) * 0.1, y_lim_max + (y_lim_max - y_lim_min) * 0.1)
        plt.show()
    return [x_range, y]


def ChaZhi_ErCi(points_2, plot=True):
    '''
    二次插值
    points_2  likes  [[3,5], [6,2], [9,10]]
    points_2  = [[3,5], [6,2], [9,10]]
    '''
    points_2 = sorted(points_2, key=lambda x: x[0])
    x_1 = points_2[0]
    x_2 = points_2[1]
    x_3 = points_2[2]
    x_range = np.linspace(x_1[0], x_3[0], 100)
    y = np.array([(x - x_2[0]) * (x - x_3[0]) / ((x_1[0] - x_2[0]) * (x_1[0] - x_3[0])) * x_1[1] +
                  (x - x_1[0]) * (x - x_3[0]) / ((x_2[0] - x_1[0]) * (x_2[0] - x_3[0])) * x_2[1] +
                  (x - x_1[0]) * (x - x_2[0]) / ((x_3[0] - x_1[0]) * (x_3[0] - x_2[0])) * x_3[1]
                  for x in x_range])
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_range, y)
        ax.plot([x_1[0], x_2[0], x_3[0]], [x_1[1], x_2[1], x_3[1]], 'o')
        x_lim_min, x_lim_max = plt.xlim()
        ax.set_xlim(x_lim_min - (x_lim_max - x_lim_min) * 0.1, x_lim_max + (x_lim_max - x_lim_min) * 0.1)
        y_lim_min, y_lim_max = plt.ylim()
        ax.set_ylim(y_lim_min - (y_lim_max - y_lim_min) * 0.1, y_lim_max + (y_lim_max - y_lim_min) * 0.1)
        plt.show()
    return [x_range, y]


if __name__ == '__main__':
    # ChaZhi_Linear([[3,5], [6,2]])
    ChaZhi_ErCi([[3,5], [6,2], [9,10]])
    pass

