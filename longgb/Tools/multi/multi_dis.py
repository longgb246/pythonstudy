#-*- coding:utf-8 -*-
from __future__ import division
import os
import sys
import cPickle
import math
import time


class myTools(object):
    @staticmethod
    def runTime(t1, name="", is_print=True):
        '''
        性能测试函数，测试运行时间。
        :param is_print: True。是否打印
        :return str 打印的字符串内容
        ex：
        t1 = time.time()
        测试的func
        runTime(t1, 'name')
        '''
        d = time.time() - t1
        min_d = math.floor(d / 60)
        sec_d = d % 60
        hor_d = math.floor(min_d / 60)
        if name != "":
            name = " ( " + name + " )"
        if hor_d >0:
            v_str = '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
        else:
            v_str = '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)
        if is_print:
            print(v_str)
        return v_str


class SubProcess(object):
    def __init__(self, dis_data, dis_res, mon_num, target):
        self._dis_data = dis_data
        self._dis_res = dis_res
        self._mon_num = mon_num
        self._split_data = None
        self._target = target
        self._args = None
        self._kwargs = None
        self._broadcast = None
        self._is_class = None
        self._all_results = []
        self._monitor(type='s')     # 进程监控

    def _monitor(self, type, run_str=''):
        '''
        进程监控
        '''
        if type == 's':
            os.system(''' echo 'kill -s 9 {0}' >>  ../process_monitor.sh '''.format(os.getpid()))
        elif type == 'r':
            os.system(''' echo '{0}' >  run_log.log; '''.format(run_str))
        elif type == 'f':
            os.system(''' echo '{0} \t Finish !' >>  ../result_monitor.log '''.format(self._mon_num))

    def readFiles(self):
        '''
        读取上游拆分的单个文件
        '''
        with open(self._dis_data, 'rb') as f:
            split_data = cPickle.load(f)
            self._split_data = split_data[0][0]
            self._broadcast = split_data[0][1:]
            self._args = split_data[1]
            self._kwargs = split_data[2]
            self._is_class = split_data[3]
            self._args = tuple(self._broadcast) + self._args

    def _runClass(self, x):
        tmp_c = self._target(x, *self._args, **self._kwargs)
        return tmp_c.run()

    def calSolver(self):
        '''
        max(500, 数据量的1%) 个数据一批的求解
        '''
        all_data_len = len(self._split_data)
        step_n = max([500, int(math.floor(all_data_len / 100))])
        split_step = int(math.ceil(all_data_len / step_n))
        t1 = time.time()
        for i in xrange(split_step):
            run_str = myTools.runTime(t1, is_print=False)
            os.system(''' echo '( {0}/{1} )  {2}'  >>  run_log.log; '''.format( i*step_n, all_data_len, run_str))
            subSplitData = self._split_data[(i*step_n) : ((i+1)*step_n)]
            if self._is_class:
                tmp_results = map(lambda x: self._runClass(x), subSplitData)
            else:
                tmp_results = map(lambda x: self._target(x, *self._args, **self._kwargs), subSplitData)
            self._all_results.extend(tmp_results)
        return self._all_results

    def saveData(self):
        '''
        保存数据
        '''
        with open(self._dis_res, 'wb') as f:
            cPickle.dump(self._all_results, f)

    def run(self):
        self._monitor(type='r', run_str='( 0/0 )')
        self.readFiles()                        # 1. 读取上游拆分的单个文件
        self.calSolver()                        # 2. 单个文件中每100个数据map出求解结果
        self.saveData()                         # 3. 保存数据结果
        self._monitor(type='f')                 # 4. 求解完毕


if len(sys.argv) > 1:
    dis_py = sys.argv[1]
    target = sys.argv[2]
    dis_data = sys.argv[3]
    dis_res = sys.argv[4]
    mon_num = sys.argv[5]
else:
    dis_py = 'test'                     # py_offline_ipc_ioa_inv_loc_cost_cal_model_classify
    target = 'test'                     # SubModelClassify
    dis_data = 'split_data.pkl'
    dis_res = 'split_result.pkl'
    mon_num = 'test'                    # dis_0


exec 'from {0} import {1} '.format(dis_py, target)


def main():
    sub_model_solve = SubProcess(dis_data, dis_res, mon_num, eval(target))
    # os.system('''echo 'PID [{0}]: {1}' >>  check.log '''.format(os.getpid(), eval(target)))
    sub_model_solve.run()


if __name__ == '__main__':
    main()

