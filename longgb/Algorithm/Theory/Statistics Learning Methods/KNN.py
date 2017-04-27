#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from string import Template
import os


def lpDistince(x1, x2, p=1):
    '''
    Lp 距离计算公式
    '''
    x1 = np.array(x1)
    x2 = np.array(x2)
    dis = np.sum(np.abs(x1-x2)**p)**(1/p) if np.ndim(x1)==1 else np.sum(np.abs(x1-x2)**p, axis=1)**(1/p)
    return dis


def lpDistinceData():
    x1 = [1, 1]
    x2 = [5, 1]
    x3 = [4, 4]
    lpDistince(x1, x3, p=1)
    lpDistince(x1, x3, p=2)
    lpDistince(x1, x3, p=3)
    lpDistince(x1, x3, p=4)
    lpDistince(x1, x2, p=1)


# http://blog.csdn.net/u010551621/article/details/44813299


class KDNode:
    def __init__(self, point=None, split=None, left = None, right = None):
        """
        KD树节点
        point : 数据点
        split : 划分域, split代表划分维度
        left : 左子节点( KDNode 对象)
        right : 右子节点( KDNode 对象)
        """
        self.point = point
        self.split = split
        self.left = left
        self.right = right


class KDTree():
    def __init__(self):
        self.root = None        # 树的根
        self.dot = None         # 画树图dot文件
        self.__index = None     # 遍历用的
        self.style = '''node [style="filled, rounded", fontname=helvetica] ;\nedge [fontname=helvetica] ;\n'''  # 画树图的风格
        self.color = ['#FF6347', '#FFA500','#FFEC8B' ,'#9AFF9A' ,'#AEEEEE' ,'#CAE1FF' ,'#B0E2FF','#AB82FF']     # 画树图的颜色列表
        self.tmp_cont = ''

    def __createKDTree(self, data_list):
        data_list = np.array(data_list)
        LEN, dimension = np.shape(data_list)                                        # 数据点的列数、维度
        if LEN == 0:
            return None
        split = np.argmax(np.var(data_list, axis=0))                                # 使用最大方差的维度，来进行划分
        data_list = data_list[np.lexsort([data_list[:, split]]), :]                 # 按照split域进行排序
        point = data_list[divmod(LEN, 2)[0], :]                                     # 选择下标为len / 2的点作为分割点
        root = KDNode(point, split)
        root.left = self.__createKDTree(data_list[0:divmod(LEN, 2)[0], :])          # 左子节点
        root.right = self.__createKDTree(data_list[(divmod(LEN, 2)[0] + 1):LEN, :]) # 右子节点
        return root

    def createKDTree(self, data_list):
        """
        创建KD树
        data_list : 数据点的集合
        return : 构造的 KDTree 的树根
        """
        self.root = self.__createKDTree(data_list)
        return self.root

    def __preOrder(self, root):
        print root.point
        if root.left:
            self.__preOrder(root.left)
        if root.right:
            self.__preOrder(root.right)

    def preOrder(self):
        '''
        前序遍历
        '''
        self.__preOrder(self.root)

    def __drawTree(self, root):
        this_index = self.__index
        self.cont += 'n{0} [label="{1}", fillcolor="{2}"];\n'.format('0'*(3-len(str(this_index)))+str(this_index),str(list(root.point)),np.random.choice(self.color))
        if root.left:
            self.__index += 1
            next_index  = self.__index
            self.cont += 'n{0} -> n{1};\n'.format('0'*(3-len(str(this_index)))+str(this_index), '0'*(3-len(str(next_index)))+str(next_index))
            self.__drawTree(root.left)
        else:               # 当发生了空的时候，为了画出是左右节点
            self.__index += 1
            next_index = self.__index
            self.tmp_cont += 'edge[color="#FFFF00", style="dashed"]; n{0} [label="", color="#FFFFFF"]'.format('0' * (3 - len(str(next_index))) + str(next_index))
            self.tmp_cont += 'n{0} -> n{1};\n'.format('0' * (3 - len(str(this_index))) + str(this_index),'0' * (3 - len(str(next_index))) + str(next_index))
            self.tmp_cont += 'edge[color=black, style="solid"];'
        if root.right:
            if self.tmp_cont != '':
                self.cont += self.tmp_cont
                self.tmp_cont = ''
            self.__index += 1
            next_index  = self.__index
            self.cont += 'n{0} -> n{1};\n'.format('0'*(3-len(str(this_index)))+str(this_index), '0'*(3-len(str(next_index)))+str(next_index))
            self.__drawTree(root.right)
        else:
            if self.tmp_cont == '':
                self.tmp_cont = None
                self.__index += 1
                next_index = self.__index
                self.cont += 'edge[color="#FFFF00", style="dashed"]; n{0} [label="", color="#FFFFFF"]'.format('0' * (3 - len(str(next_index))) + str(next_index))
                self.cont += 'n{0} -> n{1};\n'.format('0' * (3 - len(str(this_index))) + str(this_index),'0' * (3 - len(str(next_index))) + str(next_index))
                self.cont += 'edge[color=black, style="solid"];'

    def drawTree(self, is_print=True, save_path=None):
        '''
        画遍历图，数据量小可以用于分析，save_path保存dot文件，安装graphviz后，命令行使用 `dot -Tpng file.dot -o file.png`
        '''
        self.__index = 1
        self.cont = ''
        self.__drawTree(self.root)
        self.dot = 'digraph tree {{\n{0}{1} }}'.format(self.style,self.cont)
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(self.dot)
            except:
                print 'save_path is wrong!'
        if is_print:
            print self.dot


def findKnn(root, data):
    """
    root:KDTree的树根
    query:查询点
    return:返回距离data最近的点NN，同时返回最短距离min_dist
    """
    data = [2, 3, 4]

    data = np.array(data)
    root_point = root.point     # 初始化为root的节点
    min_dist = lpDistince(data, root_point, 2)

    nodeList = []               # 记录路径
    temp_root = root

    while temp_root:            # 二分查找最近的领域，直到该域叶节点
        nodeList.append(temp_root)
        dd = lpDistince(data, temp_root.point, 2)
        if min_dist > dd:       # 新计算的距离
            Knn = temp_root.point
            min_dist = dd
        ss = temp_root.split
        if data[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right

    while nodeList:             # 回溯查找
        # 使用list模拟栈，后进先出
        back_point = nodeList.pop()
        ss = back_point.split
        print "back.point = ", back_point.point
        # 判断是否需要进入父亲节点的子空间进行搜索
        if abs(data[ss] - back_point.point[ss]) < min_dist:
            if data[ss] <= back_point.point[ss]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
            if temp_root:
                nodeList.append(temp_root)
                curDist = lpDistince(data, temp_root.point)
                if min_dist > curDist:
                    min_dist = curDist
                    NN = temp_root.point
    return NN, min_dist


def KNN(list, query):
    min_dist = 9999.0
    NN = list[0]
    for pt in list:
        dist = lpDistince(query, pt)
        if dist < min_dist:
            NN = pt
            min_dist = dist
    return NN, min_dist


def getSampleData():
    '''
    获取样本数据
    '''
    data_list = [[3, 1, 4],
                 [2, 3, 7],
                 [2, 1, 3],
                 [2, 4, 5],
                 [1, 4, 4],
                 [0, 5, 7],
                 [4, 3, 4],
                 [6, 1, 4],
                 [5, 2, 5],
                 [4, 0, 6],
                 [7, 1, 6]]
    return data_list


if __name__ == '__main__':
    data_list = getSampleData()
    KDTree_ins = KDTree()
    root = KDTree_ins.createKDTree(data_list)
    KDTree_ins.preOrder()
    KDTree_ins.drawTree(save_path=r'D:\Lgb\Self\Learn\check.dot')
    os.system('D:/Lgb/Softwares/graphviz/bin/dot -Tpng D:/Lgb/Self/Learn/check.dot -o D:/Lgb/Self/Learn/check2.png')
    pass
