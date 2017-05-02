#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd


def codeVar(train_data):
    '''
    字符串编码
    '''
    all_categories = {}
    for each in train_data.columns:
        each_types = train_data[each].drop_duplicates().tolist()
        categories = {}
        for i, each_type in enumerate(each_types):
            categories[each_type] = i
        train_data[each] = train_data.loc[:, [each]].applymap(categories.get)
        all_categories[each] = categories
    all_categories = all_categories
    return train_data, all_categories


class DecisionTree():
    def __init__(self):
        self.alpha = 0.00001                # 阈值
        self.all_categories = None
        self.tree = None
        self.test_feature_name = None

    def __codeVar(self, train_data, is_test=0):
        '''
        字符串编码
        '''
        all_categories = {}
        for each in train_data.columns:
            each_types = train_data[each].drop_duplicates().tolist()
            categories = {}
            for i, each_type in enumerate(each_types):
                categories[each_type] = i
            train_data[each] = train_data.loc[:,[each]].applymap(categories.get)
            all_categories[each] = categories
        if is_test:
            pass
        else:
            self.all_categories = all_categories
        return train_data, all_categories

    def codeVar(self, train_data):
        train_data, all_categories = self.__codeVar(train_data)
        return train_data, all_categories

    def decodeVar(self, test_data):
        '''
        test数据的解码
        '''
        for each in test_data.columns:
            categories = self.all_categories[each]
            test_data[each] = test_data.loc[:, [each]].applymap(categories.get)
        return test_data

    def __printCode(self, all_categories, var_name):
        print 'Type : [ ', var_name, ' ]'
        for each_item in all_categories.items():
            print ' {0} : {1} '.format(each_item[0], each_item[1]),
        print ''

    def printCode(self, var_name=1):
        '''
        打印编码
        '''
        each_cir = self.all_categories.keys() if var_name == 1 else var_name
        for each in each_cir:
            self.__printCode(all_categories[each], each)

    def calH(self, train_data_y):
        '''
        计算信息熵
        '''
        target_type = train_data_y['y'].drop_duplicates().tolist()
        sum_res = 0
        for each in target_type:
            tmp = np.sum(train_data_y['y'] == each) / len(train_data_y)
            sum_res += - tmp * np.log2(tmp)
        return sum_res

    def infoGain(self, train_data_x, train_data_y):
        '''
        计算信息增益，返回信息增益最大的索引
        '''
        HD = self.calH(train_data_y)
        gDA = []
        for A in train_data_x.columns:
            x_type = train_data_x[A].drop_duplicates().tolist()
            HD_min = 0
            for A_v in x_type:
                this_train_data_y = train_data_y.loc[train_data_x[A] == A_v,['y']]
                HD_min += len(this_train_data_y)/len(train_data_y)*self.calH(this_train_data_y)
            gDA.append(HD - HD_min)
        return train_data_x.columns[np.argmax(gDA)], np.max(gDA)

    def __train(self, train_data_x, train_data_y, feature_name):
        Ck_type = train_data_y['y'].drop_duplicates().tolist()
        tree = {}
        if len(Ck_type) == 1:                       # 全部都是一个类
            T_type = Ck_type[0]
            tree['target_type'] = T_type
            return tree
        elif len(feature_name) == 0:                # 特征用完了，用多数的类
            T_type = Ck_type[np.argmax(map(lambda x: np.sum(train_data_y['y'] == x), Ck_type))]
            tree['target_type'] = T_type
            return tree
        else:
            infoGain_max_name, infoGain_max_v = self.infoGain(train_data_x, train_data_y)       # 计算信息增益
            if infoGain_max_v < self.alpha:         # 小于阈值，相当于特征用完了，也是用多数类
                T_type = Ck_type[np.argmax(map(lambda x: np.sum(train_data_y['y'] == x), Ck_type))]
                tree['target_type'] = T_type
                return tree
            else:
                tree['best_split'] = infoGain_max_name
                best_x_type = train_data_x[infoGain_max_name].drop_duplicates().tolist()
                split_x = map(
                    lambda x: train_data_x[train_data_x[infoGain_max_name] == x].drop(infoGain_max_name, axis=1),
                    best_x_type)
                split_y = map(lambda x: train_data_y.loc[train_data_x[infoGain_max_name] == x, ['y']], best_x_type)
                for i, each_type in enumerate(best_x_type):
                    next_tree = self.__train(split_x[i], split_y[i], split_x[i].columns.tolist())
                    tree[each_type] = next_tree
        return tree

    def train(self, train_data_x, train_data_y, feature_name, alpha=0.00001):
        self.alpha = alpha
        self.tree = self.__train(train_data_x, train_data_y, feature_name)
        return self.tree

    def __pred(self, test_x, tree):
        if tree.has_key('best_split'):
            best_split = tree['best_split']
            pre_type = self.__pred(test_x, tree[test_x[best_split].values[0]])
            return pre_type
        else:
            return tree['target_type']

    def __pred_single(self, test_x):
        '''
        预测单个
        '''
        test_x = pd.DataFrame(np.matrix(test_x), columns=self.test_feature_name)
        pre_result = self.__pred(test_x, self.tree)
        return [pre_result]

    def pred(self, test_x, feature_name):
        '''
        预测多个
        '''
        test_x = pd.DataFrame(np.matrix(test_x), columns=feature_name)
        self.test_feature_name = feature_name
        test_x_code = self.decodeVar(test_x.copy())
        result = map(self.__pred_single, test_x_code.values)
        self.pre_result = pd.DataFrame(result, columns=['target_type'])
        return pd.concat([test_x, self.pre_result], axis=1), pd.concat([test_x_code, self.pre_result], axis=1)


def getTrainData():
    '''
    获取训练样本数据
    '''
    train_data = [['青年', '否', '否', '一般', '否'],
                 ['青年', '否', '否', '好', '否'],
                 ['青年', '是', '否', '好', '是'],
                 ['青年', '是', '是', '一般', '是'],
                 ['青年', '否', '否', '一般', '否'],
                 ['中年', '否', '否', '一般', '否'],
                 ['中年', '否', '否', '好', '否'],
                 ['中年', '是', '是', '好', '是'],
                 ['中年', '否', '是', '非常好', '是'],
                 ['中年', '否', '是', '非常好', '是'],
                 ['老年', '否', '是', '非常好', '是'],
                 ['老年', '否', '是', '好', '是'],
                 ['老年', '是', '否', '好', '是'],
                 ['老年', '是', '否', '非常好', '是'],
                 ['老年', '否', '否', '一般', '否']]
    feature_name = ['age', 'work', 'house', 'credit']
    return train_data, feature_name


def getTestData():
    '''
    获取测试样本数据
    '''
    test_data = [['青年', '是', '是', '一般'],
                ['中年', '否', '是', '非常好']]
    feature_name = ['age', 'work', 'house', 'credit']
    return test_data, feature_name


if __name__ == '__main__':
    train_data, feature_name = getTrainData()
    train_data = pd.DataFrame(train_data, columns=feature_name+['y'])
    # 训练模型
    decisionTree = DecisionTree()
    train_data_code, all_categories = decisionTree.codeVar(train_data.copy())
    tree_a = decisionTree.train(train_data_code.iloc[:,range(4)], train_data_code.loc[:,['y']], feature_name)
    # 预测
    test_data, feature_name = getTestData()
    result, result_code = decisionTree.pred(test_data, feature_name)
    # 打印编码
    decisionTree.printCode()
    decisionTree.printCode(['y'])


