# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/2
  Usage   : 
"""

import multiprocessing
import Queue
from threading import Thread
from multiprocessing import Process
from subprocess import call
import argparse
import re
import yaml
import os
import time


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


# ====================== Multi Process ======================
class WorkFlow(object):
    def last_node(self):
        last = []
        for task in self.dag.values():
            if not task.on_success or len(task.on_success) == 0:
                last.append(task)
        return last

    def run(self):
        last = self.last_node()
        self.clear_state()
        self._run_in_parallel(last)

    def _run_in_parallel(self, dependencies, current=None):
        # task, dependencies task
        # start current process then return it
        # block in dependencies
        # if current action is empty then return None

        processes = {}
        for task in dependencies:
            if task.state == TaskNode.STATE_INIT:
                p = self._run_in_parallel(task.dependencies, task)
                if p:
                    processes[task.name] = p

        for name, p in processes.items():
            p.join()
            finished_task = self.dag[name]
            finished_task.complete(p.exitcode)
            # print("task: {} finished with exitcode: {}".format(finished_task.name, finished_task.exitcode))
            if p.exitcode != 0:
                # print("current task {} exit due to dependency process {} fail", current.name if current else 'root')
                exit(p.exitcode)

        if not current or not current.action or current.action == '':
            # print("task: {} finish".format(current.name if current else 'root'))
            return None
        current_process = Process(target=target_action, args=(current.action,))
        current_process.start()
        # print("task: {} start: {}".format(current.name, current.action))
        return current_process

    def clear_state(self):
        for task in self.dag.values():
            task.clear()

    def __init__(self, tasks_conf):
        self.dag = {}
        for name, conf in tasks_conf.items():
            self.dag[name] = TaskNode(name, conf)
        self.__instantiate__()

    def __repr__(self):
        return str(self.dag)

    def __instantiate__(self):
        for task in self.dag.values():
            if 'on-success' in task.conf:
                for next in task.conf['on-success']:
                    task.on_success.append(self.dag[next])
                    self.dag[next].dependencies.append(task)


class TaskNode(object):
    STATE_INIT = 0
    STATE_SUCCESS = 1
    STATE_FAIL = -1

    def clear(self):
        self.state = TaskNode.STATE_INIT
        self.exitcode = 0

    def complete(self, exitcode):
        self.exitcode = exitcode
        if exitcode == 0:
            self.state = TaskNode.STATE_SUCCESS
        else:
            self.state = TaskNode.STATE_FAIL

    def __init__(self, name, conf):
        self.name = name
        self.action = conf['action'] if 'action' in conf else None
        self.conf = conf
        self.on_success = []
        self.dependencies = []
        self.state = TaskNode.STATE_INIT
        self.exitcode = 0

    def __repr__(self):
        parents = ('(' + ''.join([task.name + '->,' for task in self.dependencies]) + ')') if len(
            self.dependencies) > 0 else ''
        children = ('(' + ''.join(['->' + task.name + ',' for task in self.on_success]) + ')') if len(
            self.on_success) > 0 else ''
        return parents + self.name + children


def target_action(cmd):
    # exitcode = call(re.split("\s+", cmd))
    # exit(exitcode)
    cmd_str = '[ PID({0}) ] {1}'.format(os.getpid(), cmd)
    print(cmd_str)
    time.sleep(3)
    print(cmd_str + '  [ Finish! ]')


def parse_workflow(file_name, i=0):
    split_func_name = 'data-split'

    with open(file_name, 'r') as f:
        wf_conf = yaml.safe_load(f)

    # just support one node workflow
    for wf_name, wf_val in wf_conf.items():
        if not 'tasks' in wf_val:
            raise KeyError('no tasks in workflow ' + wf_name)
        if 'description' in wf_val:
            # print(wf_name + ':' + wf_val['description'])
            pass
        tasks_conf = wf_val['tasks']
        break
    else:
        raise KeyError('workflow node not found')

    if split_func_name in tasks_conf.keys():
        tasks_conf.pop(split_func_name)

    update_conf(tasks_conf, i=i)
    wf = WorkFlow(tasks_conf)
    return wf


def update_conf(dict_conf, i=0):
    for k, v in dict_conf.items():
        if isinstance(v, dict):
            update_conf(v, i)
        elif k == 'action':
            dict_conf[k] = v + ' --param_dict.data_split={0}'.format(i)
    return dict_conf


def run_workflow(i):
    parser = argparse.ArgumentParser(description='forecast workflow')
    parser.add_argument('--conf', metavar='workflow.yaml', type=str, help='workflow.yaml')
    config = parser.parse_args()
    wf = parse_workflow(config.conf if config.conf else 'workflow.yaml', i)
    wf.run()


# ======================= Multi Thread ======================
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


def data_split():
    # 用于数据集的拆分，以及数据集拆分后的读取
    # 输出：输出一个 data-split 的 hdfs 文件，路径根据配置文件处理
    pass


def get_data_split():
    # 用于读取 data-split 的 hdfs 结果，生成 band-list。
    split_list = [1, 2, 3, 4, 5]
    return split_list


def get_batch_pool(split_list, parallelize=5):
    split_len = len(split_list)
    size = min(split_len, parallelize)
    pool = Pool(size=size)
    pool.add_tasks([(run_workflow, {'i': i}) for i in range(split_len)])
    return pool
    # pool.run()


def batch_workflow(file_name, parallelize=5):
    split_func_name = 'data-split'

    with open(file_name, 'r') as f:
        wf_conf = yaml.safe_load(f)

    tasks_conf = None
    for wf_name, wf_val in wf_conf.items():
        if not 'tasks' in wf_val:
            raise KeyError('no tasks in workflow ' + wf_name)
        if 'description' in wf_val:
            print(wf_name + ':' + wf_val['description'])
        tasks_conf = wf_val['tasks']

    if split_func_name in tasks_conf.keys():
        # 如果有 data-split 代表需要进行拆分。
        data_split()
        split_list = get_data_split()
        wf = get_batch_pool(split_list, parallelize=parallelize)
        pass
    else:
        wf = None
        # 如果没有，则表示可以直接走？再想想。
        pass
    return wf


def run(parallelize=5):
    parser = argparse.ArgumentParser(description='forecast workflow')
    parser.add_argument('--conf', metavar='workflow.yaml', type=str, help='workflow.yaml')
    config = parser.parse_args()
    wf = batch_workflow(config.conf if config.conf else 'workflow.yaml', parallelize=parallelize)
    wf.run()


if __name__ == '__main__':
    parallel = 3
    run(parallelize=parallel)


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

    tasks_conf = {
        'step-feature': {
            'action': 'python script_run.py component/run_feature.py src.zip component/daylevel_setting.yaml',
            'on-success': 'step-predict'
        },
        'step-promotion': {
            'action': 'python script_run.py component/run_promo_feature.py src.zip component/promo_setting.yaml',
            'on-success': 'step-predict'
        },
        'step-predict': {
            'action': 'python script_run.py component/run_predict.py src.zip component/setting-step2.yaml',
            'on-success': 'echo-finished'
        },
        'echo-finished': {
            'action': 'echo finished'
        }}
