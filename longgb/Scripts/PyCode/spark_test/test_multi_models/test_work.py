# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/2
  Usage   : 
"""

import os
import yaml
import argparse
import multiprocessing
import Queue
from threading import Thread

from workflow import run_workflow


def tmp_test():
    os.system(''' hadoop fs -get /user/mart_bca/longguangbin/sku_tmp_band_csv . ''')
    dir_list = os.listdir('sku_tmp_band_csv')
    file_list = [x for x in dir_list if x != '_SUCCESS']

    band_list = []
    for each_file in file_list:
        this_file = 'sku_tmp_band_csv' + os.sep + each_file
        with open(this_file) as f:
            band_list.extend(f.readlines())

    band_list = sorted(map(lambda x: x.replace('\n', '').strip(), band_list))


class Task(Thread):
    """ Single Task """

    def __init__(self, num, input_queue, output_queue, error_queue):
        """
        Init the Task class.
        """
        super(Task, self).__init__()
        self.thread_name = 'thread-{0}'.format(num)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.error_queue = error_queue
        self.deamon = True

    def run(self):
        """
        Run the task.
        """
        while True:
            try:
                func, kwargs = self.input_queue.get(block=False)
            except Queue.Empty:
                print '{0} finished!'.format(self.thread_name)
                break
            try:
                result = func(**kwargs)
            except Exception as e:
                # some errors
                self.error_queue.put((func.__name__, kwargs, str(e)))
            else:
                self.output_queue.put(result)


class Pool(object):
    """ Threading Pool """

    def __init__(self, size=3):
        """
        Init the Threading Pool.
        :param size: the max running threading number in the same time.
        """
        self.input_queue = Queue.Queue()
        self.output_queue = Queue.Queue()
        self.error_queue = Queue.Queue()
        self.tasks = [
            Task(i, self.input_queue, self.output_queue, self.error_queue) for i in range(size)
        ]

    def add_task(self, func, kwargs=None):
        """
        Add a single task.
        :param func: the function to run.
        :param kwargs: the arguments for the function.
        """
        kwargs = {} if kwargs is None else kwargs
        if not isinstance(kwargs, dict):
            raise TypeError('kwargs must be dict type!')
        self.input_queue.put((func, kwargs))

    def add_tasks(self, tasks):
        """
        Add batch tasks.
        :param tasks: [[func1, kwargs1], [func2, kwargs2]...]
        """
        if not isinstance(tasks, list):
            raise TypeError('tasks must be list type!')
        for func, args in tasks:
            self.add_task(func, args)

    def get_results(self):
        """
        Gets the result set.
        """
        while not self.output_queue.empty():
            print '[ Result ] : ', self.output_queue.get()

    def get_errors(self):
        """
        Gets the result set of failed execution.
        """
        while not self.error_queue.empty():
            func_name, kwargs, error_info = self.error_queue.get()
            print '[ Error ] : func: {0}, \tkwargs : {1}, \terror_info : {2}'.format(func_name, kwargs, error_info)

    def run(self):
        """
        Run the threading in the Pool.
        """
        for task in self.tasks:
            task.start()
        for task in self.tasks:
            task.join()

        self.get_results()
        self.get_errors()


def run_batch(parallelize=3):
    pool = Pool(size=parallelize)
    # pool.add_tasks([(run_workflow, (i,)) for i in range(6)])
    pool.add_tasks([(run_workflow, {'i': i}) for i in range(1)])
    pool.run()


def data_split():
    # 用于数据集的拆分，以及数据集拆分后的读取
    # 输出：输出一个 data-split 的 hdfs 文件，路径根据配置文件处理
    pass


def get_data_split():
    # 用于读取 data-split 的 hdfs 结果，生成 band-list。
    split_list = [1, 2, 3, 4, 5]
    return split_list


def _parse_workflow(file_name):
    with open(file_name, 'r') as f:
        wf_conf = yaml.safe_load(f)

    tasks_conf = None
    for wf_name, wf_val in wf_conf.items():
        if not 'tasks' in wf_val:
            raise KeyError('no tasks in workflow ' + wf_name)
        if 'description' in wf_val:
            print(wf_name + ':' + wf_val['description'])
        tasks_conf = wf_val['tasks']

    if 'data-split' in tasks_conf.keys:
        # 如果有 data-split 代表需要进行拆分。
        data_split()
        split_list = get_data_split()

        pass
    else:
        # 如果没有，则表示可以直接走？再想想。
        pass


def parse_yaml(i):
    parser = argparse.ArgumentParser(description='forecast workflow')
    parser.add_argument('--conf', metavar='workflow.yaml', type=str, help='workflow.yaml')
    config = parser.parse_args()
    wf = _parse_workflow(config.conf if config.conf else 'workflow.yaml')
    wf.run()
    return i


if __name__ == '__main__':
    # main()
    pass


def test_fail():
    function_list = [run_workflow] * 6
    print "parent process %s" % (os.getpid())

    pool = multiprocessing.Pool(3)
    for func in function_list:
        pool.apply_async(func)  # Pool执行函数，apply执行函数,当有一个进程执行完毕后，会添加一个新的进程到pool中

    print 'Waiting for all subprocesses done...'
    pool.close()
    pool.join()  # 调用join之前，一定要先调用close() 函数，否则会出错, close()执行后不会有新的进程加入到pool,join函数等待素有子进程结束
    print 'All subprocesses done.'
