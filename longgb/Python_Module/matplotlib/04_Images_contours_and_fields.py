#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 非常烦网格的，不懂，在思考一下。
def streamplot_demo_start_points():
    X, Y = (np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    # x = np.linspace(-3, 3, 100)
    # y = np.linspace(0, 0, 100)
    # y, x = np.meshgrid(y, x)
    U, V = np.mgrid[-3:3:100j, 0:0:100j]
    seed_points = np.array([[-2, 0, 1], [-2, 0, 1]])
    fig0, ax0 = plt.subplots()
    strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn, start_points=seed_points.T)
    fig0.colorbar(strm.lines)
    ax0.plot(seed_points[0], seed_points[1], 'bo')
    ax0.axis((-3, 3, -3, 3))
    plt.show()
    pass


if __name__ == '__main__':
    pass

