#-*- coding:utf-8 -*-
import numpy as np


def toSuoAMatrix(A_Matrix):
    '''
    将矩阵压缩,np.matrix -> list
    '''
    this_matrix = []
    for i, each in enumerate(np.array(A_Matrix)):
        mm = reduce(lambda x,y:x+y, map(lambda x: [i] if x==1 else [0],list(each)))
        this_matrix.append(mm)
    suoA_Matrix = list(np.sum(this_matrix, axis=0))
    return suoA_Matrix


def toAMatrix(dc_list, target):
    '''
    传入压缩的稀疏的转移矩阵序列。转化成AMatrix。
    :param target: [0, 1, 0, 1, 1, 0, 1, 1]
    :return:   [[1, 0, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 1, 1, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
    '''
    if not isinstance(target, list):
        target = list(target)
    return np.matrix(map(lambda x: map(lambda y: 1 if x == y else 0, target), range(len(dc_list))))


dc_list = ['3', '4', '5', '6', '9', '10', '316', '772']
topologyMap = [ [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1, 1, 1],       # 6 -> 5 不通的       3->2
                [0, 0, 0, 0, 0, 0, 0, 0],       # 9 撤仓              4
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]


dd = [0, 1, 3, 3, 4, 4, 6, 7]
dd = [0, 1, 5, 3, 5, 5, 6, 7]
dd_ma = toAMatrix(dc_list, dd)
np.multiply(dd_ma, np.matrix(topologyMap))
np.sum(np.multiply(dd_ma, np.matrix(topologyMap)))


is_time_out = 2 if False else 1 if False else 0

