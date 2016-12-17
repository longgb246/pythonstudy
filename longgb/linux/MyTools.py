#-*- coding:utf-8 -*-
import os
import time


def printruntime(t1):
    d = time.time() - t1
    min_d = int(d / 60)
    sec_d = d % 60
    print 'Run Time is : {0} min {1:.4f} s'.format(min_d, sec_d)

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

