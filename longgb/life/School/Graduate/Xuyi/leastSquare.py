#-*- coding:utf-8 -*-
import numpy as np  # 引入numpy
import scipy as sp
import pylab as pl
from scipy.optimize import leastsq  # 引入最小二乘函数


def real_func(x):
    '''
    目标函数
    '''
    return np.sin(2 * np.pi * x)


def fit_func(p, x):
    '''
    多项式函数
    '''
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, y, x):
    '''
    残差函数
    '''
    ret = fit_func(p, x) - y
    return ret


def residuals_func_re(p, y, x):
    '''
    正则化残差函数
    '''
    regularization = 0.1  # 正则化系数lambda
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(regularization) * p)   # 将lambda^(1/2)p加在了返回的array的后面
    return ret


# ============================== Run ==============================

def test_run():
    n = 9                                               # 多项式次数
    x = np.linspace(0, 1, 9)                            # 随机选择9个点作为x
    x_points = np.linspace(0, 1, 1000)                  # 画图时需要的连续点
    y0 = real_func(x)                                   # 目标函数
    y1 = [np.random.normal(0, 0.1) + y for y in y0]    # 添加正太分布噪声后的函数
    p_init = np.random.randn(n)                         # 随机初始化多项式参数
    plsq = leastsq(residuals_func, p_init, args=(y1, x))
    print 'Fitting Parameters: ', plsq[0]               # 输出拟合参数
    pl.plot(x_points, real_func(x_points), label='real')
    pl.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
    pl.plot(x, y1, 'bo', label='with noise')
    pl.legend()
    pl.show()


if __name__ == '__main__':
    test_run()
    pass



