#-*- coding:utf-8 -*-
# scipy 版本 0.17.1
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cubicSpline import CubicSpline


def test_sin(plot=False):
    x = np.arange(10)
    y = np.sin(x)
    cs = CubicSpline(x, y)
    xs = np.arange(-0.5, 9.6, 0.1)
    if plot:
        plt.figure(figsize=(6.5, 4))
        plt.plot(x, y, 'o', label='data')
        plt.plot(xs, np.sin(xs), label='true')
        plt.plot(xs, cs(xs), label="S")
        plt.plot(xs, cs(xs, 1), label="S'")
        plt.plot(xs, cs(xs, 2), label="S''")
        plt.plot(xs, cs(xs, 3), label="S'''")
        plt.xlim(-0.5, 9.5)
        plt.legend(loc='lower left', ncol=2)
        plt.show()
    coefficients = np.array(cs.c).T
    # coefficients.shape
    return coefficients


def test_book(plot=False):
    x = np.arange(4)
    xs = np.arange(-0.5, 4.5, 0.1)
    # y = [0.2,0,0.5,2.0,1.5,-1]
    y = [0,0.5,2.0,1.5]
    cs = CubicSpline(x, y, bc_type=((1, 0.2), (1, -1.0)))
    if plot:
        plt.figure(figsize=(6.5, 4))
        plt.plot(x, y, 'o', label='data')
        plt.plot(xs, cs(xs), label="S")
        plt.xlim(-0.5, 4.5)
        plt.legend(loc='lower left', ncol=2)
        plt.show()
    coefficients = np.array(cs.c).T
    # coefficients.shape
    return coefficients


if __name__ == '__main__':
    coefficients = test_book(plot=True)
    print coefficients
    pass
