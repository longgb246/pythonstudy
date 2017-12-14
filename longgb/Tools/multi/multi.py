#-*- coding:utf-8 -*-
import sys
import os
import math
import logging
import cPickle
import time
import re


class myTools(object):
    @staticmethod
    def mkdir(path, trash=False, clear=True):
        '''
        创建文件夹，
        :param trash: True， 表示，如果存在该文件夹，1、将该文件夹重命名为 .Trash 文件夹 2、在建立该文件夹
        '''
        if not os.path.exists(path):
            os.mkdir(path)
        elif trash:
            os.system('''cp -rf {0}  .Trash '''.format(path))
            os.system('''rm -rf {0}  '''.format(path))
            os.mkdir(path)
        if clear:
            try:
                os.system('''rm -rf .Trash  ''')
            except:
                pass
    @staticmethod
    def cp(from_path, to_path, is_dir=False):
        '''
        复制文件
        :param from_path: 原文件
        :param to_path: 复制后的文件
        '''
        if is_dir:
            os.system(''' cp -rf {0} {1} ;'''.format(from_path, to_path))
        else:
            os.system(''' cp {0} {1} ;'''.format(from_path, to_path))


class Process(object):
    def __init__(self, split_data=None, target=None, args=(), kwargs={}, logger='', dis_n=10, name='', keep_dis=False, dis_files=[]):
        self._target = target.__name__          # 目标函数
        self._split_data = split_data           # 拆分数据
        self._args = args                       # 函数参数
        self._kwargs = kwargs                   # 函数参数
        self._dis_n = dis_n                     # 进程数
        self._name = name if name != '' else 'dis_run'
        # 设置路径
        self._root_path = os.getcwd()
        self._root_dis_path = self._root_path + os.sep + self._name
        self._keep_dis = keep_dis               # 保留中间文件
        # self._dis_py = sys.argv[0].split('.')[0]
        self._dis_files = ['multi'] + dis_files
        self._dis_run_py = 'multi_dis.py'
        self._dis_py = sys.argv[0].split('.')[0] # 调用的文件
        self._dis_data = 'split_data.pkl'
        self._dis_res = 'split_result.pkl'
        self.results = None
        self._file_path = os.path.dirname(os.path.abspath(__file__))
        if logger == '':
            self._setLogger()
        else:
            self._getLogger(logger)

    def _logger(mon_str):
        def wrapper_func(func):
            def wrapper_args(self, *args, **kwargs):
                self.logger.info('{0} ...'.format(mon_str))
                result = func(self, *args, **kwargs)
                self.logger.info('{0} Finish !'.format(mon_str))
                return result
            return wrapper_args
        return wrapper_func

    def _getLogger(self, logger):
        '''
        取得log日志
        '''
        self.logger = logger

    def _setLogger(self):
        '''
        设置log日志
        '''
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(filename)s) [line:%(lineno)d] [ %(levelname)s ] %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            # filename='ModelSolve.log',
                            # filemode='w'
                            )
        self.logger = logging.getLogger("SubProcess")

    def _monitor(self, dis_n, dis_file):
        '''
        监控所有进程信息
        '''
        def findProcess(dis_path):
            '''
            监控单个进程运行状态
            '''
            dis_log = dis_path + os.sep + 'run_log.log'
            dis_pro = os.popen(''' tail -n 1 {0} '''.format(dis_log)).readlines()
            if len(dis_pro) > 0:
                dis_pro_str = dis_pro[0]
                dis_pro_str_re = re.findall('(\(.*?/.*?\))', dis_pro_str)
                log_str = dis_pro_str_re[0] if len(dis_pro_str_re) > 0 else '( --/-- )'
            else:
                log_str = '( 0/0 )'
            return log_str
        check_dis = ['dis_{0}'.format(i) for i in range(dis_n)]
        check_dis_len = dis_n + 1
        check_path = self._root_path + os.sep + dis_file
        check_file = check_path + os.sep + 'result_monitor.log'
        last_dis_len = 0
        while (check_dis_len > last_dis_len):
            check_content = os.popen(''' cat {0} '''.format(check_file)).readlines()
            last_dis_len = len(check_content)
            if last_dis_len > 1:
                complete_dis = map(lambda x: x.split('\t')[0].replace(' ', ''), check_content[1:])
                rest_dis = list(set(check_dis) - set(complete_dis))
                rest_dis_str_list = map(lambda x: x + findProcess(check_path + os.sep + x), rest_dis)
                rest_dis_len = len(rest_dis_str_list)
                rest_dis_str = ', '.join(rest_dis_str_list)
            else:
                rest_dis = check_dis
                rest_dis_str_list = map(lambda x: x + findProcess(check_path + os.sep + x), rest_dis)
                rest_dis_len = dis_n
                rest_dis_str = ', '.join(rest_dis_str_list)
            self.logger.info('[ ({0}) Dis is Running ] : {1}'.format(rest_dis_len, rest_dis_str))
            time.sleep(5)               # 每 5 秒显示一次进程状态

    @_logger('Make Monitor')
    def _mkMonitor(self):
        '''
        创建监控子进程文件
        '''
        os.system(''' echo '-------------------------------' >  {0} '''.format(self._root_dis_path + os.sep + 'result_monitor.log'))
        os.system(''' echo '#!/bin/bash' >  {0}'''.format(self._root_dis_path + os.sep + 'process_monitor.sh'))
        return 0

    @_logger('Split the Data')
    def _splitData(self):
        '''
        拆分数据
        '''
        all_data_len = len(self._split_data)
        self.split_step = int(math.ceil(all_data_len / self._dis_n))
        return 0

    @_logger('Distribute the Data')
    def _disRun(self):
        '''
        建立分布文件。启动多个进程。
        '''
        myTools.mkdir(self._root_dis_path, trash=True)
        self._mkMonitor()
        for i in xrange(self._dis_n):
            sub_path = self._root_dis_path + os.sep + 'dis_{0}'.format(i)
            myTools.mkdir(sub_path)
            tmp_split = [self._split_data[i*self.split_step: (i+1)*self.split_step],
                         # self._target,
                         self._args,
                         self._kwargs]
            myTools.cp(self._root_path + os.sep + '{0}.py'.format(self._dis_py), sub_path + '/.')
            myTools.cp(self._file_path + os.sep + self._dis_run_py, sub_path + '/.')
            myTools.cp(self._file_path + os.sep + '__init__.py', sub_path + '/.')
            for each in self._dis_files:
                is_dir = os.path.isdir(self._root_path + os.sep + each)
                myTools.cp(self._root_path + os.sep + each, sub_path + '/.', is_dir=is_dir)
            with open(sub_path + os.sep + self._dis_data, 'wb') as f:
                cPickle.dump(tmp_split, f)
            self._disRunPython(sub_path, 'dis_{0}'.format(i))
            self.logger.info('Run the Sub Process [ {0} ]'.format(i))
        return 0

    def _disRunPython(self, sub_path, mon_num):
        '''
        启动python文件处理
        '''
        os.chdir(sub_path)
        os.system(''' nohup python {0} {1} {2} {3} {4} {5} >  nohup_multi_{5}.log  2>&1 &'''.format(
                      self._dis_run_py, self._dis_py, self._target, self._dis_data, self._dis_res, mon_num))
        os.chdir(self._root_path)

    @_logger('Read Dis Files')
    def _readDisFiles(self):
        '''
        读取分布的结果文件
        '''
        dis_res = []
        for each in os.listdir(self._root_dis_path):
            if os.path.isdir(self._root_dis_path + os.sep + each):
                this_dis_path = self._root_dis_path + os.sep + each + os.sep + self._dis_res
                with open(this_dis_path, 'rb') as f:
                    this_dis_data = cPickle.load(f)
                    dis_res.extend(this_dis_data)
        self.results = dis_res
        return self.results

    @_logger('Clean Dis Files')
    def _cleanDisFiles(self):
        os.system(''' rm -rf {0} '''.format(self._name))
        return 0

    def start(self, dis_n=-1):
        self._dis_n = dis_n if dis_n > 0 else self._dis_n
        self._splitData()                           # 2. 拆分数据
        self._disRun()                              # 3. 启动多个进程
        self._monitor(self._dis_n, self._name)      # 启动进程监控
        self._readDisFiles()                        # 4. 读取分布式数据
        if not self._keep_dis:
            self._cleanDisFiles()
        return self.results

