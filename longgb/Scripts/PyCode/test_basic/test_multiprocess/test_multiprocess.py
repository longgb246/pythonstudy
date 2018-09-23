# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/15
  Usage   : 
"""

import os
import multiprocessing


def run_workflow():
    pass


def test_fail():
    function_list = [run_workflow] * 6
    print "parent process %s" % (os.getpid())

    pool = multiprocessing.Pool(3)
    for func in function_list:
        pool.apply_async(func)  # Pool执行函数，apply执行函数,当有一个进程执行完毕后，会添加一个新的进程到pool中

    print 'Waiting for all subprocesses done...'
    pool.close()
    pool.join()  # 调用join之前，一定要先调用close() 函数，否则会出错, close()执行后不会有新的进程加入到pool,join函数等待素有子进程结束
    print 'All subprocesses done.'

    tasks_conf = {
        'step-feature': {
            'action': 'python script_run.py component/run_feature.py src.zip component/daylevel_setting.yaml',
            'on-success': 'step-predict'
        },
        'step-promotion': {
            'action': 'python script_run.py component/run_promo_feature.py src.zip component/promo_setting.yaml',
            'on-success': 'step-predict'
        },
        'step-predict': {
            'action': 'python script_run.py component/run_predict.py src.zip component/setting-step2.yaml',
            'on-success': 'echo-finished'
        },
        'echo-finished': {
            'action': 'echo finished'
        }}
