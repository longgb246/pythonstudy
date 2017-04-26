#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import copy


class perceptronClass():
    '''
    感知器模型，预测的 y 应该是 +1（正例） or -1（负例）
    '''
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.w = 0
        self.b = 0
        self.record = []
        self.record_w = []
        self.record_b = []
        self.cir_num = 0

    def reset(self, alpha=1):
        self.alpha = alpha

    def pred(self, pre_x):
        w_pre = self.w.reshape(-1, 1)
        pre_x = np.array(pre_x).reshape(-1, 2)
        return np.sign(pre_x.dot(w_pre)+self.b)

    def train(self, data_x, data_y):
        self.w = np.zeros(len(data_x.columns))
        self.record = []
        self.record_w = []
        self.record_b = []
        self.cir_num = 0
        z = True
        n_cir = 100
        while np.any(z) and n_cir > 0:
            n_cir -= 1
            z = data_y * (data_x.dot(self.w).reshape(3, -1) + self.b) <= 0
            index = np.argmax(z['y']) if np.any(z) else -1
            if index != -1:
                self.w += self.alpha * data_x.iloc[index, :].values * data_y.iloc[index, :].values
                self.b += self.alpha * data_y.iloc[index, :].values
            else:
                break
            self.record.append(index + 1)
            self.record_w.append(copy.deepcopy(self.w))
            self.record_b.append(copy.deepcopy(self.b))
        return [self.w, self.b, self.cir_num]


def getTestData():
    '''
    获取测试用的数据
    '''
    x = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    data = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
    x_cols = ['x1', 'x2']
    y_cols = ['y']
    data.columns = x_cols + y_cols
    return data, x_cols, y_cols


if __name__ == '__main__':
    data, x_cols, y_cols = getTestData()
    perModel = perceptronClass()
    aa_arg = perModel.train(data.loc[:, x_cols], data.loc[:, y_cols])
    perModel.pred(data.loc[:, x_cols])
    pass

