#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from collections import defaultdict


class NativeBayes():
    def __init__(self):
        pass
    pass


def train(train_data_x, train_data_y, alpha=1):
    BayesProb = defaultdict(float)
    SubBayesProb = defaultdict(lambda: defaultdict(float))
    train_data_x, train_data_y = train_data.loc[:, ['x1', 'x2']], train_data['y']
    train_data_x = pd.DataFrame(train_data_x, columns=['x'+str(x+1) for x in range(np.shape(train_data_x)[1])])
    train_data_y = pd.DataFrame(train_data_y, columns=['y'])
    target_type = train_data_y.drop_duplicates()
    K = len(target_type)
    N = len(train_data_x)
    for each in target_type.values:
        BayesProb[each[0]] = (np.sum(train_data_y['y']==each[0]) + alpha)/(N + K * alpha)



def getTrainData():
    '''
    获取训练样本数据
    '''
    train_data = [[1, 'S', -1],
                 [1, 'M', -1],
                 [1, 'M', 1],
                 [1, 'S', 1],
                 [1, 'S', -1],
                 [2, 'S', -1],
                 [2, 'M', -1],
                 [2, 'M', 1],
                 [2, 'L', 1],
                 [2, 'L', 1],
                 [3, 'L', 1],
                 [3, 'L', 1],
                 [3, 'M', 1],
                 [3, 'M', 1],
                 [3, 'L', -1]]
    return train_data


def getTestData():
    '''
    获取测试样本数据
    '''
    test_data = [2, 'S']
    return test_data


if __name__ == '__main__':
    train_data = getTrainData()
    test_data = getTestData()
    train_data = pd.DataFrame(train_data, columns=['x1', 'x2', 'y'])
    train(train_data.loc[:,['x1','x2']], train_data['y'])
    pass
