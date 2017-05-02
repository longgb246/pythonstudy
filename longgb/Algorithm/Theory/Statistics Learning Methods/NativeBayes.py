#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from collections import defaultdict


class NativeBayes():
    def __init__(self):
        self.BayesProb = defaultdict(float)
        self.target_type = None
        self.pre_result = None

    def getIndex(self, y, x, x_v):
        '''
        y ：代表 y 的值。
        x ：代表第几个 x 若为 -1 表示不用 x 的概率。
        x_v ：代表该 x 的值。
        '''
        return ','.join([y, x, x_v])

    def train(self, train_data_x, train_data_y, alpha=1):
        '''
        训练模型
        '''
        self.BayesProb = defaultdict(float)
        train_data_x = pd.DataFrame(train_data_x, columns=['x'+str(x+1) for x in range(np.shape(train_data_x)[1])])
        train_data_y = pd.DataFrame(train_data_y, columns=['y'])
        self.target_type = train_data_y['y'].drop_duplicates().tolist()
        K = len(self.target_type)
        N = len(train_data_x)
        x_type = map(lambda x: train_data_x[x].drop_duplicates().tolist(),train_data_x.columns)
        for c_k in self.target_type:
            self.BayesProb[self.getIndex(str(c_k), '-1', '-1')] = (np.sum(train_data_y['y'] == c_k) + alpha) / (N + K * alpha)
            for j, each_x in enumerate(x_type):
                Sj = len(each_x)
                for each_x_j in each_x:
                    self.BayesProb[self.getIndex(str(c_k), str(j), str(each_x_j))] = (np.sum((train_data_y['y'] == c_k)&(train_data_x.iloc[:,j] == each_x_j)) + alpha)/(np.sum(train_data_y['y'] == c_k) + Sj * alpha)

    def __pred(self, test_x):
        '''
        预测数据
        '''
        max_p = 0
        for c_k in self.target_type:
            p_this = 1
            p_this *= self.BayesProb[self.getIndex(str(c_k), '-1', '-1')]
            for j, each_x in enumerate(test_x):
                p_this *= self.BayesProb[self.getIndex(str(c_k), str(j), str(each_x))]
            if (p_this > max_p):
                max_p = p_this
                test_y = c_k
        return [test_y, max_p]

    def pred(self, test_x):
        test_x = pd.DataFrame(np.matrix(test_x))
        result = map(self.__pred, test_x.values)
        self.pre_result = pd.DataFrame(result, columns=['target_type', 'prob'])
        return self.pre_result


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
    # test_data = [2, 'S']
    test_data = [[2, 'S'],[3, 'M']]
    return test_data


if __name__ == '__main__':
    train_data = getTrainData()
    test_data = getTestData()
    train_data = pd.DataFrame(train_data, columns=['x1', 'x2', 'y'])
    nativeBayes = NativeBayes()
    nativeBayes.train(train_data.loc[:,['x1','x2']], train_data['y'])
    pre_result = nativeBayes.pred(test_data)
    print pre_result

