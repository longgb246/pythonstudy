# -*- coding:utf-8 -*-
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# https://github.com/kawache/Python-B-spline-examples
def Drawing():
    '''
    画曲线，造了一个向量。
    '''
    ctr = np.array([(3, 1), (2.5, 4), (0, 1), (-2.5, 4), (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1), ])
    x = ctr[:, 0]
    y = ctr[:, 1]
    l = len(x)
    t = np.linspace(0, 1, l - 2, endpoint=True)
    t = np.append([0, 0, 0], t)
    t = np.append(t, [1, 1, 1])
    tck = [t, [x, y], 3]
    u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)
    out = interpolate.splev(u3, tck)
    plt.plot(x, y, 'k--', label='Control polygon', marker='o', markerfacecolor='red')
    plt.plot(out[0], out[1], 'b', linewidth=2.0, label='B-spline curve')
    plt.legend(loc='best')
    plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
    plt.title('Cubic B-spline curve evaluation')
    plt.show()


def Calcul():
    ctr = np.array([(3, 1), (2.5, 4), (0, 1), (-2.5, 4), (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])
    x = ctr[:, 0]
    y = ctr[:, 1]
    tck, u = interpolate.splprep([x, y], k=3, s=0)          # k=3使用三次B样条，s表示平滑程度。见参考文档
    u = np.linspace(0, 1, num=50, endpoint=True)
    out = interpolate.splev(u, tck)
    plt.figure()
    plt.plot(x, y, 'ro', out[0], out[1], 'b')
    plt.legend(['Points', 'Interpolated B-spline', 'True'], loc='best')
    plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
    plt.title('B-Spline interpolation')
    plt.show()


# wrong
def test():
    ctr = np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, 3), (9, -1), (11, 3), (13, 3)])
    ctrs = [ctr, np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, 0), (9, -1), (11, 3), (13, 3)])]
    ctrs.append(np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, -1), (9, -1), (11, 3), (13, 3)]))
    for ctr in ctrs:
        x = ctr[:, 0]
        y = ctr[:, 1]
        tck, u = interpolate.splprep([x, y], k=3)
        u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u, tck)
        # plt.figure()
        plt.plot(x, y, 'ro', out[0], out[1], 'b')
        plt.legend(['Points', 'Interpolated B-spline', 'True'], loc='best')
        plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
        plt.title('B-Spline interpolation')
        # plt.show()


# http://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
def try2():
    import scipy.interpolate as si
    # points = [[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]]
    ctr = np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, 3), (9, -1), (11, 3), (13, 3)])
    ctrs = [ctr, np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, 0), (9, -1), (11, 3), (13, 3)])]
    ctrs.append(np.array([(0, 0), (2, 4), (4, 3), (3, -1), (6, -2), (9, -1), (11, 3), (13, 3)]))
    fig = plt.figure()
    for points in ctrs:
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        t = range(len(points))
        ipl_t = np.linspace(0.0, len(points) - 1, 100)
        # models，仅仅一个轴？
        x_tup = si.splrep(t, x, k=3)
        y_tup = si.splrep(t, y, k=3)
        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
        x_i = si.splev(ipl_t, x_list)
        y_i = si.splev(ipl_t, y_list)
        # ==============================================================================
        # Plot
        # ==============================================================================
        plt.plot(x, y, '-og')
        plt.plot(x_i, y_i, 'r')
        plt.xlim([min(x) - 0.3, max(x) + 0.3])
        plt.ylim([min(y) - 0.3, max(y) + 0.3])
        plt.title('Splined f(x(t), y(t))')
    plt.show()
    i = 1
    if i == 1:
        fig = plt.figure()
        ax = fig.add_subplot(231)
        plt.plot(t, x, '-og')
        plt.plot(ipl_t, x_i, 'r')
        plt.xlim([0.0, max(t)])
        plt.title('Splined x(t)')

        ax = fig.add_subplot(232)
        plt.plot(t, y, '-og')
        plt.plot(ipl_t, y_i, 'r')
        plt.xlim([0.0, max(t)])
        plt.title('Splined y(t)')

        ax = fig.add_subplot(233)
        plt.plot(x, y, '-og')
        plt.plot(x_i, y_i, 'r')
        plt.xlim([min(x) - 0.3, max(x) + 0.3])
        plt.ylim([min(y) - 0.3, max(y) + 0.3])
        plt.title('Splined f(x(t), y(t))')

        ax = fig.add_subplot(234)
        for i in range(7):
            vec = np.zeros(11)
            vec[i] = 1.0
            x_list = list(x_tup)
            x_list[1] = vec.tolist()
            x_i = si.splev(ipl_t, x_list)
            plt.plot(ipl_t, x_i)
        plt.xlim([0.0, max(t)])
        plt.title('Basis splines')
        plt.show()


if __name__ == '__main__':
    # Drawing()
    try2()
    pass

