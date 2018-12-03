# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/3
"""  
Usage Of 'suibian' : 
"""

import os

date_list = ['2018-07-01', '2018-07-05', '2018-07-20', '2018-08-01', '2018-08-22',
             '2018-09-07', '2018-09-11', '2018-10-01', '2018-10-10', '2018-10-30', '2018-11-13']


def exec_run(dt_v):
    exec_str = '''python script_run.py runPPandas.py src.zip setting.yaml {dt}'''.format(dt=dt_v)
    print(exec_str)
    os.system(exec_str)
    print(exec_str + ' Finish !')


if __name__ == '__main__':
    for dt in date_list:
        try:
            exec_run(dt)
        except:
            error_str = '''echo '{dt} is error !' > error_log.log '''.format(dt=dt)
            os.system(error_str)
