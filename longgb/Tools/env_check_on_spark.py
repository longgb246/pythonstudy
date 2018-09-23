# -*- coding: utf-8 -*-
'''
@author: longguangbin
@contact: longguangbin@163.com

用于检测哪些 python 包没有在集群节点上正确安装
通过 spark-submit 来提交这个脚本
'''

from pyspark import SparkContext, SparkConf


def mapTest(models, import_str):
    '''
    Map for testing whether the package is installed correctly.
    '''
    import sys as sys
    import socket
    no_c = 0
    test_res = []
    no_list = []
    sys_m = sys.modules
    for each_model in models:
        is_wrong = 0
        s_str = import_str.format(package=each_model)
        exec(s_str)
        test_res.append(is_wrong)
        if is_wrong == 1:
            no_c += 1
            no_list.append(each_model)
    if no_c == 0:
        return [[0]]
    else:
        ip = socket.gethostbyname(socket.gethostname())             # get the ip address
        return [[1, no_c, str(no_list), ip]]


def collectResult(s):
    '''
    Collect the mapping results.
    '''
    map = {}
    for rs in s:
        if int(rs[0]) == 1:
            ev = eval(rs[2])
            ip = rs[3]
            for r_1 in ev:
                if ip not in map:
                    map[ip] = []
                if r_1 not in map[ip]:
                    map[ip].append(r_1)
    lens = len(map)
    fs = ''
    for ip in map.keys():
        fs += '{0} : {1}\n'.format(ip, str(map[ip]))
    return lens, fs


import_str = '''try:
    import {package}
    sys_m['{package}']
except:
    is_wrong = 1
'''

# Functions to be test.
models = [  'numpy', 'pandas', 'scipy', 'hyperopt', 'xgboost', 'sklearn',
            'theano', 'keras', 'nltk', 'pytest', 'virtualen']


if __name__ == '__main__':
    conf = SparkConf().setAppName("fdc_predict_train")
    sc = SparkContext(conf=conf)
    # Map the func
    s = sc.parallelize(range(10000), 100).mapPartitions(lambda x: mapTest(models, import_str)).collect()
    lens, fs = collectResult(s)
    if lens == 0:
        print("[ Success ] python包安装正确！")
    else:
        print("[ Error ] 有" + str(lens) + "个节点出现少包的情况：\n" + fs)

