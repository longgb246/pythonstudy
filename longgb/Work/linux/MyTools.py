#-*- coding:utf-8 -*-
import os
import numpy as np

import time
def printruntime(t1):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d >0:
        print 'Run Time is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d)
    else:
        print 'Run Time is : {0} min {1:.4f} s'.format(min_d, sec_d)


def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

