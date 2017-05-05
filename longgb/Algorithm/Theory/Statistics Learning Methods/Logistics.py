#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd


class Logistics():
    def __init__(self):
        self.y_type = None
        self.w = None

    def sigmoid(self, w, x):
        '''
        logistics 函数
        '''
        w = np.array(w)
        x = np.array(x)
        return 1/(1+np.exp(-x.dot(w)))

    def __gradientDescent(self, train_data_x, train_data_y, alpha=1):
        '''
        梯度下降算法， 假设我的算法是对的，
        '''
        train_data_y = np.array(train_data_y['y'].values)
        w = np.array([1] * len(train_data_x.columns))
        j = np.array([1] * len(train_data_x.columns))
        m = len(train_data_x)
        train_data_x = np.array(train_data_x)
        n_cir = 100
        j_list = []
        while (n_cir > 0) and (np.max(np.abs(j))>0.0001):
            n_cir -= 1
            y_cal = self.sigmoid(w, train_data_x)
            error = y_cal - train_data_y
            j = alpha / m * np.sum(error.reshape(-1, 1) * train_data_x, axis=0)
            w = w - j
            j_list.append(j)
        return w

    def __multiClass(self, train_data_x, train_data_y, alpha=1):
        '''
        多分类 one vs rest 的方法。
        '''
        y_type = list(np.unique(train_data_y['y']))
        w_list = []
        for each in y_type:                 # 对每一个类建模
            this_train_data_y = train_data_y.copy()
            this_train_data_y.loc[this_train_data_y['y'] != each] = -2
            this_train_data_y.loc[this_train_data_y['y'] == each] = -1
            this_train_data_y.loc[this_train_data_y['y'] == -1] = 1
            this_train_data_y.loc[this_train_data_y['y'] == -2] = 0
            w_list.append(self.__gradientDescent(train_data_x, train_data_y, alpha=alpha))
        return w_list

    def train(self, train_data_x, train_data_y, alpha=1):
        self.y_type = list(np.unique(train_data_y['y']))
        if len(self.y_type) == 2:
            self.w = self.__gradientDescent(train_data_x, train_data_y, alpha=alpha)
        elif len(self.y_type) > 2:
            self.w = self.__multiClass(train_data_x, train_data_y, alpha=alpha)
        else:
            print 'Number of Type y is less than or equal to one!'
            return
        return self.w

    def __pred(self, w, data_x):
        '''
        二分类预测
        '''
        data_x = np.array(data_x)
        prob = self.sigmoid(w, data_x)
        return prob

    def __predMulti(self, data_x):
        '''
        多分类预测
        '''
        prob_list = []
        for i in range(len(self.y_type)):
            prob = self.__pred(self.w[i], data_x)
            prob_list.append(prob)
        return prob_list

    def pred(self, data_x):
        '''
        预测
        '''
        if len(self.y_type) == 2:
            self.prob = self.__pred(self.w, data_x)
        elif len(self.y_type) > 2:
            prob = self.__predMulti(data_x)
            prob = np.array(prob)
            self.prob = prob.T
        return self.prob


def getTrainData():
    '''
    获取训练样本数据 + 截距均为1
    '''
    train_data = [[1, 17, 1, 1], [1, 68, 1, 0], [0, 17, 0, 0],
                  [1, 44, 0, 0], [1, 18, 1, 0], [0, 45, 0, 1],
                  [1, 48, 1, 0], [1, 68, 0, 0], [0, 44, 0, 1],
                  [1, 55, 0, 0], [1, 48, 1, 1], [0, 67, 0, 0],
                  [1, 75, 1, 1], [1, 17, 0, 0], [0, 55, 0, 1],
                  [0, 35, 0, 1], [1, 70, 1, 1], [1, 61, 1, 0],
                  [0, 42, 1, 1], [1, 72, 1, 0], [1, 19, 1, 2],
                  [0, 57, 0, 0], [1, 35, 0, 1], [1, 69, 0, 2],
                  [0, 28, 0, 1], [1, 19, 1, 0], [1, 23, 1, 1],
                  [0, 20, 0, 1], [1, 62, 1, 0], [1, 19, 0, 0],
                  [0, 38, 1, 0], [0, 39, 1, 1], [1, 72, 1, 1],
                  [0, 45, 0, 1], [0, 40, 1, 1], [1, 74, 1, 0],
                  [0, 47, 1, 1], [0, 55, 0, 0], [1, 31, 0, 1],
                  [0, 52, 0, 0], [0, 68, 0, 1], [1, 16, 1, 0],
                  [0, 55, 0, 1], [0, 25, 1, 0], [1, 61, 1, 0]]
    var_name = ['eye', 'age', 'drive', 'y']
    train_data_tmp = pd.DataFrame([1]*len(train_data),columns=['intercept'])
    train_data = pd.concat([train_data_tmp, pd.DataFrame(train_data, columns=var_name)], axis=1)
    return train_data


def getTestData():
    '''
    获取测试样本数据
    '''
    test_data = [['青年', '是', '是', '一般'],
                ['中年', '否', '是', '非常好']]
    feature_name = ['age', 'work', 'house', 'credit']
    return test_data, feature_name


def dataset_fixed_cov():
    '''
    形成 2 个 Gaussians 样本使用相同的方差矩阵
    '''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    train_data_tmp = pd.DataFrame([1] * len(train_data), columns=['intercept'])
    data = pd.concat([pd.DataFrame(X, columns=[''])])
    return X, y


if __name__ == '__main__':
    train_data = getTrainData()
    logistics = Logistics()
    w = logistics.train(train_data.drop(['y'], axis=1), train_data.loc[:,['y']])
    prob = logistics.pred(train_data.drop(['y'], axis=1))
    # 形成高斯分布
    train_data = dataset_fixed_cov()

