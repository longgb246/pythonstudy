# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/3
"""  
Usage Of 'suibian' : 
"""

import os

date_list = [
    '2018-10-10', '2018-10-30', '2018-11-13', '2018-09-12', '2018-09-13', '2018-09-14',
    '2018-09-15', '2018-09-16', '2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20',
    '2018-09-21', '2018-09-22', '2018-09-23', '2018-09-24', '2018-09-25'
]


def exec_run(dt_v):
    exec_str = '''python script_run.py runPPandas.py src.zip setting.yaml {dt}'''.format(dt=dt_v)
    print(exec_str)
    res_v = os.system(exec_str)
    print(exec_str + ' Finish !')
    return res_v


if __name__ == '__main__':
    for dt in date_list:
        try:
            res = exec_run(dt)
        except:
            error_str = '''echo '{dt} is error !' >> error_log.log '''.format(dt=dt)
            os.system(error_str)
            res = -1
        if res != 0:
            error_str = '''echo '{dt} is error !' >> error_log.log '''.format(dt=dt)
            os.system(error_str)
        else:
            error_str = '''echo '{dt} is Success !' >> success_log.log '''.format(dt=dt)
            os.system(error_str)
