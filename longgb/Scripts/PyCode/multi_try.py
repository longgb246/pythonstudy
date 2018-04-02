#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import sys
import time
import logging


from multi.multi import Process


from MyTools import myTools


def aa(x):
    return x + 3


def main():
    print 'get Data'
    data = range(10000000)
    # data = map(lambda x: [x], data)

    map(lambda x: aa(x), data)

    print 'Method 1 ...'
    t1 = time.time()
    sub = Process(split_data=data, target=aa, dis_n=10, keep_dis=True, dis_files=['MyTools.py'])
    results2 = sub.start()
    myTools.runTime(t1)
    print len(results2)


    print 'Method 2 ...'
    t1 = time.time()
    results = map(lambda x: aa(x), data)
    myTools.runTime(t1)
    print len(results)


if __name__ == '__main__':
    main()


# import multiprocessing
# multiprocessing.Process
