#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd


def lpDistince(x1, x2, p=1):
    '''
    Lp 距离计算公式
    '''
    x1 = np.array(x1)
    x2 = np.array(x2)
    dis = np.sum(np.abs(x1-x2)**p)**(1/p)
    return dis


def lpDistinceData():
    x1 = [1, 1]
    x2 = [5, 1]
    x3 = [4, 4]
    lpDistince(x1, x3, p=1)
    lpDistince(x1, x3, p=2)
    lpDistince(x1, x3, p=3)
    lpDistince(x1, x3, p=4)
    lpDistince(x1, x2, p=1)


if __name__ == '__main__':
    pass
