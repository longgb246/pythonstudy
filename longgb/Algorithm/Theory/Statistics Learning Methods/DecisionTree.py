#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import os
from copy import deepcopy


class DecisionTree():
    def __init__(self):
        self.alpha = 0.00001                # 阈值
        self.all_categories = None
        self.tree = None
        self.test_feature_name = None
        self.dot = None
        self.deep = None
        self.method = None
        self.alpha = 0.8
        self.Ck_type = None
        self.style = '''node [shape=box, style="filled, rounded", fontname="SimSun"] ;\nedge [fontname="SimSun"] ;\n'''  # 画树图的风格
        self.color = ['#EEE9E9', '#FFE4C4', '#FFEC8B', '#9AFF9A', '#AEEEEE', '#CAE1FF', '#B0E2FF', '#AB82FF']  # 画树图的颜色列表

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

    def __decodeVar(self, var_name):
        categories = self.all_categories[var_name]
        return dict([v, k] for k,v in categories.iteritems())

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

    def infoGain(self, train_data_x, train_data_y, method='id3'):
        '''
        计算信息增益，返回信息增益最大的索引
        '''
        HD = self.calH(train_data_y)
        num_y = len(train_data_y)
        V1 = num_y*HD
        gDA = []
        gDA_rate = []
        V2_list = []
        n_i = []
        for A in train_data_x.columns:
            x_type = train_data_x[A].drop_duplicates().tolist()
            HD_min = 0
            tmp_V2_list = []
            i = 0
            for A_v in x_type:
                i += 1
                this_train_data_y = train_data_y.loc[train_data_x[A] == A_v,['y']]
                num_y2 = len(this_train_data_y)
                HD_min += len(this_train_data_y)/len(train_data_y)*self.calH(this_train_data_y)
                tmp_V2_list.append(self.calH(this_train_data_y)*num_y2)
            gDA.append(HD - HD_min)
            gDA_rate.append((HD - HD_min)/HD)
            V2_list.append(tmp_V2_list)
            n_i.append(i)
        # 计算是否剪枝
        HD_split = V2_list[np.argmax(gDA)]
        HD_rate_split = V2_list[np.argmax(gDA_rate)]
        n_i_split = n_i[np.argmax(gDA)] - 1
        n_i_rate_split = n_i[np.argmax(gDA_rate)] - 1
        V_split = np.sum(HD_split) + self.alpha*n_i_split - V1
        V_rate_split = np.sum(HD_rate_split) + self.alpha*n_i_rate_split - V1
        gDA_list = [train_data_x.columns[np.argmax(gDA)], np.max(gDA), HD, V_split]
        gDA_rate_list = [train_data_x.columns[np.argmax(gDA_rate)], np.max(gDA_rate), HD, V_rate_split]
        if method == 'id3':
            return gDA_list
        else:
            return gDA_rate_list

    def __cartSplit(self, data_x, data_y):
        '''
        变量空间的划分
        '''
        def calGini(x):
            num_x = len(data)
            data_ge = data[data['x'] > x]
            data_le = data[data['x'] <= x]
            gini_ge_min = 0
            gini_le_min = 0
            for each in data_y_unique:
                data_ge_y = data_ge[data_ge['y'] == each]
                data_le_y = data_le[data_le['y'] == each]
                gini_ge_min += (len(data_ge_y) / len(data_ge)) ** 2 if len(data_ge) > 0 else 0
                gini_le_min += (len(data_le_y) / len(data_le)) ** 2 if len(data_le) > 0 else 0
            gini_ge = 1 - gini_ge_min
            gini_le = 1 - gini_le_min
            gini = len(data_ge) / num_x * gini_ge + len(data_le) / num_x * gini_le
            return gini

        data = pd.concat([pd.DataFrame(data_x), pd.DataFrame(data_y)], axis=1)
        data.columns = ['x', 'y']
        data_y_unique = np.unique(data_y).tolist()
        data_x_unique = np.unique(data_x).tolist()
        data_x_split = [data_x_unique[0] - (data_x_unique[1] - data_x_unique[0]) / 2] + \
                       [data_x_unique[i] + (data_x_unique[i + 1] - data_x_unique[i]) / 2 for i in
                        range(len(data_x_unique) - 1)] + \
                       [data_x_unique[-1] + (data_x_unique[-1] - data_x_unique[-2]) / 2]
        ginis = map(calGini, data_x_split)
        best_split = data_x_split[np.argmin(ginis)]
        return np.min(ginis), best_split

    def __cartSplitBest(self, train_data_x, train_data_y, feature_name):
        this_train_data_y = train_data_y['y'].values.tolist()
        ginis_list = []
        best_split_list = []
        for each in feature_name:           # 计算每一个特征的gini指数
            this_train_data_x = train_data_x[each].values.tolist()
            ginis, best_split = self.__cartSplit(this_train_data_x, this_train_data_y)
            ginis_list.append(ginis)
            best_split_list.append(best_split)
        return feature_name[np.argmin(ginis_list)], np.min(ginis_list)

    def __train(self, train_data_x, train_data_y, feature_name, method):
        '''
        ID3 和 C4.5 算法
        '''
        Ck_type = train_data_y['y'].drop_duplicates().tolist()          # 目标变量的分类数
        tree = {}
        if len(Ck_type) == 1:                       # 全部都是一个类
            T_type = Ck_type[0]
            tree['target_type'] = T_type
            tree['sample'] = len(train_data_y)
            if method == 'CART':
                tree['Gini'] = 0
            else:
                HD = self.calH(train_data_y)
                tree['HD'] = HD
            return tree
        elif len(feature_name) == 0:                # 特征用完了，用多数的类
            T_type = Ck_type[np.argmax(map(lambda x: np.sum(train_data_y['y'] == x), Ck_type))]
            tree['target_type'] = T_type
            tree['sample'] = len(train_data_y)
            if method == 'CART':
                this_gini = 1
                gini_min = 0
                for each in self.Ck_type:
                    gini_min += len(train_data_y[train_data_y['y'] == each])/len(train_data_y)
                this_gini -= gini_min
                tree['Gini'] = this_gini
            else:
                HD = self.calH(train_data_y)
                tree['HD'] = HD
            return tree
        else:
            if method == 'CART':
                select_var_name, select_var_v = self.__cartSplitBest(train_data_x.copy(), train_data_y.copy(), feature_name)
            else:
                select_var_name, select_var_v, HD, V = self.infoGain(train_data_x, train_data_y, method=method)       # 计算信息增益、信息增益比

            print 'select_var_name', select_var_name
            print 'select_var_v', select_var_v

            if (select_var_v < self.alpha) and (method != 'CART'):         # 小于阈值，相当于特征用完了，也是用多数类
                T_type = Ck_type[np.argmax(map(lambda x: np.sum(train_data_y['y'] == x), Ck_type))]
                tree['target_type'] = T_type
                tree['sample'] = len(train_data_y)
                if method == 'CART':
                    tree['Gini'] = select_var_v
                else:
                    tree['HD'] = HD
                    tree['V'] = V
                return tree
            else:
                T_type = Ck_type[np.argmax(map(lambda x: np.sum(train_data_y['y'] == x), Ck_type))]
                tree['target_type'] = T_type
                tree['best_split'] = select_var_name
                tree['sample'] = len(train_data_y)
                if method == 'CART':
                    tree['Gini'] = select_var_v
                else:
                    if method == 'id3':
                        tree['info_gain'] = select_var_v
                    else:
                        tree['info_gain_rate'] = select_var_v
                    tree['HD'] = HD
                    tree['V'] = V
                best_x_type = train_data_x[select_var_name].drop_duplicates().tolist()
                split_x = map(lambda x: train_data_x[train_data_x[select_var_name] == x].drop(select_var_name, axis=1),best_x_type)
                split_y = map(lambda x: train_data_y.loc[train_data_x[select_var_name] == x, ['y']], best_x_type)
                for i, each_type in enumerate(best_x_type):
                    next_tree = self.__train(split_x[i], split_y[i], split_x[i].columns.tolist(), method)
                    tree[each_type] = next_tree
        return tree

    def __cutTree(self, tree):
        '''
        ID3 和 C4.5 算法 的剪枝
        '''
        if tree.has_key('V'):
            V = tree['V']
            if V > 0:   # 剪枝
                sub_list = tree.keys()
                key_remove = ['HD', 'sample', 'target_type']
                for each in key_remove:
                    sub_list.remove(each)
                for each in sub_list:
                    tree.pop(each)
            else:       # 不剪枝
                sub_list = tree.keys()
                gain = 'info_gain' if self.method == 'id3' else 'info_gain_rate'
                key_remove = ['HD', 'V', 'best_split', 'sample', 'target_type'] + [gain]
                for each in key_remove:
                    sub_list.remove(each)
                for each in sub_list:
                    self.__cutTree(tree[each])
        else:
            return

    def train(self, train_data_x, train_data_y, feature_name, alpha=0.00001, method='id3', is_cut=True):
        self.alpha = alpha
        self.method = method
        self.Ck_type = train_data_y['y'].drop_duplicates().tolist()
        if method in ['id3', 'c4.5', 'CART']:
            self.tree = self.__train(train_data_x, train_data_y, feature_name, method=method)
        else:
            print 'The [ {0} ] method of train model is wrong ! You shuld input "id3", "c4.5" or "CART" !'.format(method)
        self.__cutTree(self.tree)           # 剪枝
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

    def __plotTree(self, tree, this_index, deep, method):
        if self.deep is not None:   # 控制画树的深度
            if deep > self.deep:
                return
        best_split = tree.pop('best_split') if tree.has_key('best_split') else ''
        if method == 'id3':
            info_gain = tree.pop('info_gain') if tree.has_key('info_gain') else ''
        elif method == 'c4.5':
            info_gain_rate = tree.pop('info_gain_rate') if tree.has_key('info_gain_rate') else ''
        else:
            gini = tree.pop('Gini') if tree.has_key('Gini') else ''
        sample = tree.pop('sample') if tree.has_key('sample') else ''
        if method == 'CART':
            pass
        else:
            HD = tree.pop('HD') if tree.has_key('HD') else ''
        key_list = tree.keys()
        remove_list = ['V', 'target_type']
        for each in remove_list:
            if each in key_list:
                key_list.remove(each)
        if best_split != '':
            categories = self.__decodeVar(best_split)
            if method == 'id3':
                str_need = r'best_split : {0}\ninfo_gain : {1:.4f}\nsample : {2}\nHD : {3:.4f}'.format(best_split, info_gain, sample, HD)
            elif method == 'c4.5':
                str_need = r'best_split : {0}\ninfo_gain_rate : {1:.4f}\nsample : {2}\nHD : {3:.4f}'.format(best_split, info_gain_rate, sample, HD)
            else:
                str_need = r'best_split : {0}\nGini : {1:.4f}\nsample : {2}'.format(best_split, gini, sample)
            self.cont += 'n{0} [label="{1}", fillcolor="{2}"];\n'.format('0'*(3-len(str(this_index)))+str(this_index), str_need, np.random.choice(self.color))
            if self.deep is not None:
                if deep == self.deep:
                    pass
                else:
                    for each in key_list:
                        self.__index += 1
                        self.cont += 'n{0} ;\n'.format('0'*(3-len(str(self.__index)))+str(self.__index))
                        self.cont += 'n{0} -> n{1} [label="{2}"] ;\n'.format('0'*(3-len(str(this_index)))+str(this_index), '0'*(3-len(str(self.__index)))+str(self.__index), categories[each])
                        self.__plotTree(deepcopy(tree[each]), self.__index, deep+1, method)
            else:
                for each in key_list:
                    self.__index += 1
                    self.cont += 'n{0} ;\n'.format('0' * (3 - len(str(self.__index))) + str(self.__index))
                    self.cont += 'n{0} -> n{1} [label="{2}"] ;\n'.format('0' * (3 - len(str(this_index))) + str(this_index), '0' * (3 - len(str(self.__index))) + str(self.__index), categories[each])
                    self.__plotTree(deepcopy(tree[each]), self.__index, deep + 1, method)
        else:
            self.__index += 1
            target_type = tree['target_type']
            categories = self.__decodeVar('y')
            if method=='CART':
                str_need = 'target_type : {0}\nsample : {1}\nGini : {2:.4f}'.format(categories[target_type], sample, gini)
            else:
                str_need = 'target_type : {0}\nsample : {1}\nHD : {2:.4f}'.format(categories[target_type], sample, HD)
            self.cont += 'n{0} [label="{1}", fillcolor="{2}"];\n'.format('0'*(3-len(str(this_index)))+str(this_index), str_need, np.random.choice(self.color))

    def plotTree(self, is_print=True, save_path=None, plot_deep=None):
        '''
        打印生成的树
        is_print: 是否打印 dot 文件
        save_path: 保存路径
        plot_deep: 打印树的最大深度
        '''
        self.__index = 1
        self.cont = ''
        self.deep = plot_deep
        self.__plotTree(deepcopy(self.tree), self.__index, 0, method=self.method)
        self.dot = 'digraph tree {{\n{0}{1} }}'.format(self.style, self.cont)
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(self.dot)
            except:
                print 'save_path is wrong!'
        if is_print:
            print self.dot


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
    decisionTree = DecisionTree()
    train_data_code, all_categories = decisionTree.codeVar(train_data.copy())

    # 1、id3 训练模型
    tree_id3 = decisionTree.train(train_data_code.iloc[:,range(4)], train_data_code.loc[:,['y']], feature_name, is_cut=True)
    # 预测
    test_data, feature_name = getTestData()
    result, result_code = decisionTree.pred(test_data, feature_name)
    print result
    # 画图
    save_path = r'D:\Lgb\Self\Learn\workspace\decisiontree_id3.dot'
    decisionTree.plotTree(save_path=save_path, is_print=False)
    os.system('D:/Lgb/Softwares/graphviz/bin/dot -Tpng D:/Lgb/Self/Learn/workspace/decisiontree_id3.dot -o D:/Lgb/Self/Learn/workspace/decisiontree_id3.png')

    # 2、c4.5 训练模型
    tree_c45 = decisionTree.train(train_data_code.iloc[:,range(4)], train_data_code.loc[:,['y']], feature_name, method='c4.5', is_cut=True)
    # 预测
    test_data, feature_name = getTestData()
    result, result_code = decisionTree.pred(test_data, feature_name)
    print result
    # 画图
    save_path = r'D:\Lgb\Self\Learn\workspace\decisiontree_c45.dot'
    decisionTree.plotTree(save_path=save_path, is_print=False)
    os.system('D:/Lgb/Softwares/graphviz/bin/dot -Tpng D:/Lgb/Self/Learn/workspace/decisiontree_c45.dot -o D:/Lgb/Self/Learn/workspace/decisiontree_c45.png')

    # 3、CART 算法
    # 这里写的 CART 算法有一点问题，原应该是将区域进行划分，然后拆分每个区域，这里仅仅是找到划分后的区域计算的最佳值的特征，然后根据该特征有的分类数，进行下一步迭代，即不是二分类。
    # 之后在改，因为这里写的框架有点乱了。比较混乱，之后调整好在改过来。
    tree_CART = decisionTree.train(train_data_code.iloc[:, range(4)], train_data_code.loc[:, ['y']], feature_name, method='CART', is_cut=True)
    # 预测
    test_data, feature_name = getTestData()
    result, result_code = decisionTree.pred(test_data, feature_name)
    # print result
    # 画图
    save_path = r'D:\Lgb\Self\Learn\workspace\decisiontree_cart.dot'
    decisionTree.plotTree(save_path=save_path, is_print=False)
    os.system('D:/Lgb/Softwares/graphviz/bin/dot -Tpng D:/Lgb/Self/Learn/workspace/decisiontree_cart.dot -o D:/Lgb/Self/Learn/workspace/decisiontree_cart.png')

    # # 打印编码
    # decisionTree.printCode()
    # decisionTree.printCode(['y'])

