#-*- coding:utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D             # 3d 库
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_2d_to_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')           # 设置 3d 的画质, 产生一个3d的坐标
    # =========================== 1、画出 z 轴 sin 图 ===========================
    x = np.linspace(0, 1, 100)
    y = np.sin(x * 2 * np.pi) / 2 + 0.5
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')      # zs=0 是 z 的坐标，zdir='z' 表示在 z 轴
    # =========================== 2、画出 y 轴 散点图 ===========================
    colors = ['r', 'g', 'b', 'k']
    x = np.random.sample(20*len(colors))    # sample 取出均匀分布[0,1]之间的样本
    y = np.random.sample(20*len(colors))
    c_list = []
    for c in colors:
        c_list.append([c]*20)
    ax.scatter(x, y, zs=0, zdir='y', c=c_list ,label='points in (x, z)')
    # =========================== 3、调整图片 ====================================
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)
    plt.show()




