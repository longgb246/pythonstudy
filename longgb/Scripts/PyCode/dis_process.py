#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import os
import pandas as pd
from multiprocessing import Process, Queue
import time


class npl():
    '''
    numpy local method : npl
    '''
    @staticmethod
    def loc(np_array, cols):
        '''
        按照cols的布尔序列选出array的某些行
        cols 为 [True, False] 组成的 list 或者 np.array
        使用方法如下：
        np_loc(test, test[:,-1] == 1)  # 抽出test最后一列为1的np.array
        '''
        return np_array[np.where(cols)[0], :]
    @staticmethod
    def sort(np_array, cols, ascending=[]):
        '''
        按cols从小到大排列array的行数据
        cols 为 [3, 1] 数字组成的 list
        使用方法如下：
        np_sort(test, [3, 1])           # 按照test的第4列，第2列从小到大排序
        '''
        if ascending==[]:
            sort_data = np_array[:, cols[::-1]].T
        else:
            sort_data = ([map(lambda x: 1 if x else -1, ascending[::-1])] * np_array[:, cols[::-1]]).T
        return np_array[np.lexsort(sort_data), :]

def findModel(data):
    '''
    [第二版]
    data：第0列是index，第1列是sales，第2列是std，第3列是labels
    '''
    def _findAModel(data_1):
        def _getIndex(data_1, x):
            target_loc = npl.sort(data_1[:x, :], [1], ascending=[False])[0, 0]
            return zip(data_1[:, 0], [target_loc]*x + list(data_1[x:, 0]))
        # data_1 = npl.loc(data, data[:, -1] == 1)
        this_len = data_1.shape[0]
        data_1 = npl.sort(data_1, [2], ascending=[False])
        return map(lambda x: _getIndex(data_1, x), range(1, this_len+1))
    labels_uniq = np.unique(data[:, -1])
    tmp_v = reduce(lambda m, n: [m1 + n1 for m1 in m for n1 in n], map(lambda x: _findAModel(npl.loc(data, data[:, -1] == x)), labels_uniq))
    result = map(lambda m: map(lambda x: np.int(x), npl.sort(np.array(m), [0])[:, 1]), tmp_v)
    return result

def multiModels(multi_data, result_q, mon_q, num_i):
    print 'Start num_i[{0}] : PID[{1}]'.format(num_i, os.getpid())
    result = []
    len_num = len(multi_data)
    for i, each in enumerate(multi_data):
        sku = each[0]
        data = each[1]
        this_result = findModel(data)
        result.append([sku, this_result])
        mon_q.put('[{2}]( {0}/{1} )'.format(i, len_num, num_i))
    result_q.put(result, block=True)

def getData():
    index_data = range(8)
    sales = np.random.rand(8) * 8
    std = np.random.rand(8) * 2
    labels = [0, 1, 0, 1, 2, 0, 2, 1]
    data = np.array(zip(index_data, sales, std, labels))
    splitData = []
    for i in range(10):
        tmpData = []
        for j in range(1000):
            tmpData.append(['sku_{0:04.0f}'.format(i*1000+j), data])
        splitData.append(tmpData)
    return splitData

def main(logger):
    logger.info('Start dis main : PID[{0}]'.format(os.getpid()))
    logger.info('Read Data ...')
    # print 'Start dis main : PID[{0}]'.format(os.getpid())
    # print 'Read Data ...'
    splitData = getData()
    logger.info('Read Data Finish ! ')
    logger.info('Process ...')
    # print 'Read Data Finish ! '
    # print 'Process ...'
    div_n = 10
    result_list = []
    result_q = Queue()
    mon_q_lis = []
    multi_pools = []
    mon_pro_list = ['( 0/0 )']*div_n
    for i in range(div_n):
        mon_q = Queue()
        multi_pools.append(Process(target = multiModels, args = (splitData[i], result_q, mon_q, i)))
        mon_q_lis.append(mon_q)
    for each in multi_pools:
        each.start()
    is_alive = len(multi_pools)
    logger.info('Process Finish ! ')
    # print 'Process Finish ! '

    while is_alive > 0:
        last_alive = 0
        for i, each in enumerate(multi_pools):
            this_is_alive = each.is_alive()
            if this_is_alive:
                last_alive += 1
            tmp_mon_pro = mon_pro_list[i]
            while True:
                try:
                    tmp_mon_pro = mon_q_lis[i].get(block=False)
                except:
                    break
            mon_pro_list[i] = tmp_mon_pro if this_is_alive else '100%'
        if last_alive < is_alive:
            is_alive = last_alive
        while True:
            try:
                this_data = result_q.get(block=False)
                result_list += this_data
            except:
                break
        # print 'is_alive : {0}'.format(is_alive), str(mon_pro_list)
        logger.info('is_alive : {0} {1}'.format(is_alive, str(mon_pro_list)))
        time.sleep(1)

    # print 'result_list[0] : ', result_list[0]
    # print 'result_list[1] : ', result_list[1]
    # print 'result_list[2] : ', result_list[2]
    # print len(result_list)
    logger.info('result_list[0] : {0}'.format(result_list[0]))
    logger.info('result_list[0] : {0}'.format(result_list[1]))
    logger.info('result_list[0] : {0}'.format(result_list[2]))
    logger.info(len(result_list))
    return result_list


if __name__ == '__main__':
    main()


# 1767334.0 / 1618918.0
# 1.0916760453586902
# 2 / 1.0916760453586902 = 1.8320453292925953  ~ 1.832
