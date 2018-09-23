#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: "longguangbin"
@contact: longguangbin@163.com

logger 自定义日志
'''

from string import Template
import time
import datetime

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window


def date_diff(x):
    '''
    Datetime.timedelta diff between 2 dates .
    '''
    if len(x) > 1:
        dt1 = datetime.datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S")
        delta_dt = (dt2 - dt1).__str__()
    else:
        delta_dt = '0:00:00'
    return delta_dt


class Logger(object):
    def __init__(self, logger='', time_format='asctime', logger_sc=None, spark=None):
        '''
        Init Logger Class
        '''
        self.getOrCreate(logger)
        self.format_str = Template('${time_str} ( ${class_name_str} ) [ ${levelname_str} ] ${message_info}')
        self.time_format = self._timeFormat(time_format)
        self.class_name_str = ''
        self.record_format = '%Y-%m-%d %H:%M:%S'
        self.lock = 0
        self.logger_sc = logger_sc
        self.spark = spark
        self.model_str = 'No Model Parsed !'
    @staticmethod
    def loggerDef(mon_str):
        def wrapper_func(func):
            def wrapper_args(self, *args, **kwargs):
                self.class_name_str = func.__name__ if hasattr(func, '__name__') else func.__class__ if hasattr(func, '__class__') else func.__module__ if hasattr(func, '__module__') else ''
                self.logger_info('{0} Start !'.format(mon_str))
                try:
                    result = func(self, *args, **kwargs)
                except (SystemExit, KeyboardInterrupt) as e:
                    e_name = e.__class__ if hasattr(e, '__class__') else 'SystemExit or KeyboardInterrupt'
                    e_message = e.message if hasattr(e, 'message') else ''
                    self.logger_warning('{0} {1} : {2} !'.format(mon_str, e_name, e_message))
                    self.lock += 1
                    if self.lock == 1:
                        self.saveLogger()
                    raise
                except Exception as e:
                    e_name = e.__class__ if hasattr(e, '__class__') else 'Exception'
                    e_message = e.message if hasattr(e, 'message') else ''
                    self.logger_error('{0} {1} : {2} !'.format(mon_str, e_name, e_message))
                    self.lock += 1
                    if self.lock == 1:
                        self.saveLogger()
                    raise
                self.logger_info('{0} Finish !'.format(mon_str))
                return result
            return wrapper_args
        return wrapper_func
    def _message(self, levelname_str='INFO', class_name_str='', message_info=''):
        '''
        all level message
        '''
        this_time = time.localtime(time.time())
        time_str = time.strftime(self.time_format, this_time)
        record_time_str = time.strftime(self.record_format, this_time)
        message = self.format_str.substitute(time_str=time_str, class_name_str=class_name_str, levelname_str=levelname_str, message_info=message_info)
        self._recordMessage(levelname_str, class_name_str, message_info, record_time_str)
        return message
    def logger_debug(self, message_info=''):
        '''
        DEBUG message
        '''
        print self._message(levelname_str='DEBUG', class_name_str=self.class_name_str, message_info=message_info)
    def logger_info(self, message_info=''):
        '''
        INFO message
        '''
        print self._message(levelname_str='INFO', class_name_str=self.class_name_str, message_info=message_info)
    def logger_warning(self, message_info=''):
        '''
        WARNING message
        '''
        print self._message(levelname_str='WARNING', class_name_str=self.class_name_str, message_info=message_info)
    def logger_error(self, message_info=''):
        '''
        ERROR message
        '''
        print self._message(levelname_str='ERROR', class_name_str=self.class_name_str, message_info=message_info)
    def _recordMessage(self, levelname_str='', class_name_str='', message_info='', record_time_str=''):
        '''
        record this message
        '''
        self.logger.append([levelname_str, class_name_str, message_info, record_time_str])
    def _timeFormat(self, time_format='asctime'):
        '''
        set time format
        '''
        if time_format == 'asctime':
            time_str = '%a, %d %b %Y %H:%M:%S'
        elif time_format == 'normal_time':
            time_str = '%Y-%m-%d %H:%M:%S'
        else:
            time_str = time_format
        return time_str
    def _getLogger(self, logger):
        '''
        get logger object
        '''
        self.logger = logger
    def _setLogger(self):
        '''
        create logger record object
        '''
        self.logger = []
    def getOrCreate(self, logger=''):
        '''
        get or create logger record object
        '''
        if (logger == '') or (logger is None):
            self._setLogger()
        else:
            self._getLogger(logger=logger)
    def saveLogger(self, table_name='', dt='', logger_sc=None, spark=None):
        '''
        Save the logger information
        '''
        logger_sc = self.logger_sc if logger_sc is None else logger_sc
        spark = self.spark if spark is None else spark
        if logger_sc is not None:
            logger_sp = logger_sc.parallelize(self.logger).toDF(['info_level', 'class_name', 'message', 'datetime'])
        elif spark is not None:
            logger_sp = spark.createDataFrame(self.logger, ['info_level', 'class_name', 'message', 'datetime'])
        else:
            logger_sp = SparkSession.builder.getOrCreate().createDataFrame(self.logger, ['info_level', 'class_name', 'message', 'datetime'])
        # logger_sp.registerTempTable('_logger_insert_table')
        # self.spark.sql(''' insert overwrite table {0} partition (dt='{1}') select * from _logger_insert_table '''.format(table_name, dt))
        # logger_sp.show(10000)
        windowspec_r = Window.orderBy('datetime').rowsBetween(Window.currentRow, 1)
        logger_order_sp = logger_sp.select('*', F.udf(date_diff)(F.collect_list('datetime').over(windowspec_r)).alias('date_diff'), F.lit(self.model_str).alias('Models')).orderBy(F.col('date_diff').desc())
        # Show the Logger Result
        print '\n\n\n\n\n======================== Show the Logger Result ========================\n\n'
        logger_order_sp.show(20)
        print '\n'.join(map(lambda x: '\t'.join(map(str, list(x))), logger_order_sp.limit(20).collect()))
        # print str(logger_order_sp.limit(20).collect()).replace('Row', '\nRow')
        print '\n\n\n\n\n======================== Show the Logger Warning, Debug And Error ========================\n\n'
        logger_order_check_sp = logger_order_sp.where(''' info_level in ('DEBUG', 'WARNING', 'ERROR') ''').orderBy('datetime')
        logger_order_check_sp.show(1000)
        print '\n'.join(map(lambda x: '\t'.join(map(str, list(x))), logger_order_check_sp.collect()))
        # print str(logger_order_check_sp.collect()).replace('Row', '\nRow')
        self.lock += 1
        return logger_sp
    def loggerGetModel(self, model_str):
        self.model_str = model_str


def example():
    from pyspark.sql import functions as F
    from pyspark.sql import Window
    from collections import Counter
    from scipy import interpolate
    import re
    import os
    import sys
    import numpy as np

    class processor_base(Logger):
        '''
        数据预处理的基类，用于抽取各个子类都可用到的模块
        当ds==None时，表示跟时间无关的预处理
        '''
        def __init__(self,raw_data, key=None, ds=None, logger=None, spark=None):
            """Parameters
            raw_data : the input data frame contain the target values
            """
            super(processor_base, self).__init__(logger=logger, spark=spark)
            self.raw_data = raw_data
            self.key = key
            self.ds = ds
        def _validate_inputs(self,col_list):
            """Validates the inputs to DataInput.
              check the input column list must be the subset of the raw_data
            """
            if not set(col_list).difference(self.raw_data.columns):
                print 'Columns is ok,Begin to Run....'
            else:
                raise ValueError('''The columns not in data's columns ''')
        def _validate_column_names(self,ban_str):
            """Validates the name of a seasonality, holiday, or regressor.
            Parameters"""
            if ban_str in self.raw_data.columns:
                raise ValueError('Name cannot contain %s'.format(ban_str))
            reserved_names =['key','ds']
            if self.key == 'key':
                reserved_names.remove('key')
            if self.ds == 'ds' or self.ds is None:
                reserved_names.remove('ds')
            for name in reserved_names:
                if name in self.raw_data.columns:
                    raise ValueError('Name "{}" is reserved.'.format(name))
        def _reset_col(self,ban_str):
            self._validate_column_names(ban_str)
            self.raw_data = self.raw_data.withColumnRenamed(self.key, 'key')
            if self.ds is not None:
                self.raw_data = self.raw_data.withColumnRenamed(self.ds, 'ds')
            return self
        def _recover_col(self,data):
            '''
            针对新生成的数据集key和ds 是否需要和原来的一直，如果一致，则需要列名恢复
            '''
            data = data.withColumnRenamed('key',self.key)
            if self.ds is not None:
                data = data.withColumnRenamed('ds',self.ds)
            return data

    class transform(processor_base):
        '''
        (self,raw_data,key,ds,agg_func,_value,mid_colnames,end_value = Window.currentRow):
        '''
        def __init__(self, raw_data, key=None, ds=None, trans_method=None, target_columns=None, logger=None, spark=None):
            '''
            Parameters
            :param raw_data: the input data frame contain the target values
            :param key: the key of data
            :param ds: the dt
            '''
            super(transform, self).__init__(raw_data, key=key, ds=ds, logger=logger, spark=spark)
            self.func_name_list = ['log', 'exp', 'sqrt', 'pow']                 # Supported function methods.
            self.target_columns = target_columns
            self.trans_method = trans_method
        @Logger.loggerDef('_checkArgs')
        def _checkArgs(self, trans_method):
            '''
            Check and Arrange the Argument : 'trans_method'
            '''
            num_str_list = re.findall(r'([\d|\.]+)', trans_method)              # Extract the letter portion of the 'trans_method'.
            arg_str_list = re.findall(r'([a-z]+)', trans_method)                # Extract the number portion of the 'trans_method'.
            num_str = num_str_list[0] if len(num_str_list)==1 else 'no_value'
            arg_str = arg_str_list[0] if len(arg_str_list)==1 else 'value_error'
            if arg_str in self.func_name_list:
                func = getattr(F, arg_str)                                      # Taking the function of F.
            else:
                raise ValueError('''The Argument 'trans_method' Must Be One Of Those: 'log', 'exp', 'sqrt', 'pow'. ''')
            return func, num_str, arg_str
        @Logger.loggerDef('_transform')
        def _transform(self):
            self.logger_warning('fdsafdsf')
            func, num_str, arg_str = self._checkArgs(self.trans_method)
            if arg_str == 'pow':
                num_str = 2 if num_str == 'no_value' else num_str                                           # The default parameter for setting 'pow' is 2.
                trans_list = map(lambda x: func(F.col(x), F.lit(float(num_str))).alias(arg_str + '{0}_'.format(num_str) + x), self.target_columns)
            elif arg_str == 'log':
                if num_str == 'no_value':
                    trans_list = map(lambda x: func(F.col(x)).alias(arg_str + '_' + x), self.target_columns)     # Means the logarithm of base 'e'
                else:
                    trans_list = map(lambda x: func(float(num_str), F.col(x)).alias(arg_str + '{0}_'.format(num_str) + x), self.target_columns)
            else:
                trans_list = map(lambda x: func(F.col(x)).alias(arg_str + '_' + x), self.target_columns)
            return trans_list
        @Logger.loggerDef('transformData')
        def transformData(self):
            '''
            Specifies the transformation of the specified column on the original data set and returns the data set.
            :param target_columns: the target column to transform
            :param trans_method: 'exp', 'sqrt', 'log', 'pow'
            :return: sparkDataFrame
            (annotation) : 'log', 'pow' support those formats: 'logm.n', 'powm.n'.
                            (1) 'log2.3'(Means the logarithm of base 2.3);  (2)'pow3.3'(Means 3.3 power);
                            (3)'log'(Means the logarithm of base 'e');      (4)'pow'(Means square)
            '''
            trans_list = self._transform()
            data_columns = self.raw_data.columns
            return self.raw_data.select(data_columns + trans_list)

    class imputer(processor_base):
        def __init__(self, raw_data, key=None, ds=None, logger=None, spark=None):
            '''
            Parameters
            :param raw_data: the input data frame contain the target values
            :param key: the key of data
            :param ds: the dt
            '''
            super(imputer, self).__init__(raw_data, key=key, ds=ds, logger=logger, spark=spark)
            self.number_impute_method_list = ['first', 'last', 'mode', 'random', 'max', 'min', 'mean', 'median', 'linear', 'quadratic', 'cubic']
            self.string_impute_method_list = ['first', 'last', 'mode', 'random']
            self.func_type = []
            self.order_func = ['linear', 'quadratic', 'cubic', 'first', 'last']
        @Logger.loggerDef('_checkArgs')
        def _checkArgs(self, target_columns, layer, impute_method, order_by):
            '''
            Check and Arrange the Argument : 'target_columns', 'layer', 'impute_method'
            '''
            data_columns = self.raw_data.columns
            if (not isinstance(target_columns, list)) or (len(target_columns)<1):
                raise ValueError(''' You Must Specify The 'target_columns' To Be Split By . Your 'target_columns' : {0} '''.format(target_columns))
            for each_column in target_columns:
                if each_column not in data_columns:
                    raise ValueError(''' The Value Of 'target_columns' ({0}) Is Not In Your Data Columns . '''.format(each_column))
            for each_layer in layer:
                if each_layer not in data_columns:
                    raise ValueError(''' The Value Of 'layer' ({0}) Is Not In Your Data Columns . '''.format(each_layer))
            if impute_method in self.string_impute_method_list:                     # 'func_type' is used for checking whether the function is matching the type of columns
                self.func_type.append('string')
            if impute_method in self.number_impute_method_list:
                self.func_type.append('number')
            else:
                raise ValueError(''' The 'impute_method' should be in : {0} '''.format(str(self.number_impute_method_list).replace('[', '').replace(']', '')))
            if (impute_method in self.order_func) and (order_by==[]):
                raise ValueError(''' The 'impute_method' is an ordered method, you must specify the 'order_by'. ''')
            return target_columns, layer, impute_method, order_by
        @Logger.loggerDef('_impute')
        def _impute(self, layer, impute_method, target_columns, order_by):
            def _imputeMethod(data, impute_method, func_type, target_columns, order_func, origin_len):
                '''
                The function used for rdd's map function
                '''
                def _findNull(x):
                    '''
                    Find the Null Value
                    '''
                    if (x is None) or (x==np.nan) or (x==np.inf) or (x==-np.inf):
                        return False
                    if isinstance(x, str) and ((str(x).strip()=='') or (str(x).strip().lower()=='null')):
                        return False
                    if isinstance(x, unicode) and ((str(x).strip()=='') or (str(x).strip().lower()=='null')):
                        return False
                    return True
                def _interpolateMethod(sub_data, valid_data, valid_index, no_valid_index):
                    '''
                    Interpolate Method : 'linear', 'quadratic', 'cubic'
                    '''
                    if no_valid_index[0]==0:
                        valid_index = [0] + valid_index
                        valid_data = [valid_data[0]] + valid_data
                    if (no_valid_index[-1]==(len(sub_data)-1)) and (no_valid_index[-1]!=0):
                        valid_index = valid_index + [len(sub_data)-1]
                        valid_data = valid_data + [valid_data[-1]]
                    func = func_map[impute_method]
                    f = func(valid_index, valid_data, kind=impute_method)
                    imputed_value = f(range(len(sub_data)))
                    return imputed_value
                def _statsMethod(sub_data, valid_data, not_null):
                    '''
                    Stats Method : 'max', 'min', 'mean', 'median'
                    '''
                    func = func_map[impute_method]
                    impute_value = func(valid_data)
                    imputed_value = np.where(not_null, sub_data, [impute_value]*len(sub_data))
                    return imputed_value
                def _firstMethod(sub_data, valid_data, no_valid_index):
                    '''
                    First Method : 'first'
                    '''
                    for impute_index in no_valid_index:
                        if impute_index <= 0:
                            sub_data[impute_index] = valid_data[0]
                        else:
                            sub_data[impute_index] = sub_data[impute_index-1]
                    imputed_value = sub_data
                    return imputed_value
                def _lastMethod(sub_data, valid_data, no_valid_index):
                    '''
                    Last Method : 'last'
                    '''
                    for impute_index in no_valid_index[::-1]:
                        if impute_index >= (len(sub_data)-1):
                            sub_data[impute_index] = valid_data[-1]
                        else:
                            sub_data[impute_index] = sub_data[impute_index+1]
                    imputed_value = sub_data
                    return imputed_value
                def _modeMethod(sub_data, valid_data, not_null):
                    '''
                    Mode Method : 'mode'
                    '''
                    impute_value = sorted(Counter(valid_data).items(), key=lambda x: x[1])[0][0]
                    imputed_value = np.where(not_null, sub_data, [impute_value]*len(sub_data))
                    return imputed_value
                def _randomMethod(sub_data, valid_data, not_null):
                    '''
                    Random Method : 'random'
                    '''
                    impute_value = np.random.choice(valid_data, 1).tolist()[0]
                    imputed_value = np.where(not_null, sub_data, [impute_value]*len(sub_data))
                    return imputed_value
                def _subImputeMethod(sub_data, impute_method, column_name):
                    '''
                    The function used for each target column's map function
                    '''
                    not_null = map(lambda x: _findNull(x), sub_data)            # Finding the Null value, True/False list
                    valid_data = filter(_findNull, sub_data)                     # The Not Null value
                    valid_index = filter(_findNull, np.where(not_null, range(len(sub_data)), [None]*len(sub_data)).tolist())    # The index of Not Null value
                    no_valid_index = sorted(list(set(range(len(sub_data))) - set(valid_index)))
                    column_type = 'number' if isinstance(valid_data[0], float) or isinstance(valid_data[0], int) else 'string'
                    imputed_value = 'impossibleValue'
                    if column_type not in func_type:
                        raise TypeError(''' The Column '{0}' Type is : {1}, the method '{2}' not match . '''.format(column_name, column_type, impute_method))
                    if impute_method in ['linear', 'quadratic', 'cubic']:
                        imputed_value = _interpolateMethod(sub_data, valid_data, valid_index, no_valid_index)
                    if impute_method in ['max', 'min', 'mean', 'median']:
                        imputed_value = _statsMethod(sub_data, valid_data, not_null)
                    if impute_method == 'first':
                        imputed_value = _firstMethod(sub_data, valid_data, no_valid_index)
                    if impute_method == 'last':
                        imputed_value = _lastMethod(sub_data, valid_data, no_valid_index)
                    if impute_method == 'mode':
                        imputed_value = _modeMethod(sub_data, valid_data, not_null)
                    if impute_method == 'random':
                        imputed_value = _randomMethod(sub_data, valid_data, not_null)
                    imputed_value = map(str, imputed_value) if column_type == 'string' else map(float, imputed_value)
                    return imputed_value
                if impute_method in order_func:                 # Sort by the specified column.
                    data_sort = sorted(data, key=lambda x: x[-1])
                    origin_keep_data = map(lambda x: x[:origin_len], data_sort)
                    data_cal = map(lambda x: x[origin_len:-1], data_sort)
                else:
                    origin_keep_data = map(lambda x: x[:origin_len], data)
                    data_cal = map(lambda x: x[origin_len:], data)
                func_map = {'max': np.max, 'min': np.min, 'mean': np.mean, 'median': np.median, 'linear': interpolate.interp1d, 'quadratic': interpolate.interp1d, 'cubic': interpolate.interp1d}
                data_T = map(list, zip(*data_cal))              # Transpose the list
                data_imputed_T = map(lambda i: _subImputeMethod(data_T[i], impute_method, target_columns[i]), range(len(data_T)))
                data_imputed_tmp = map(list, zip(*data_imputed_T))
                data_imputed = map(lambda i: origin_keep_data[i]+data_imputed_tmp[i], range(len(data_imputed_tmp)))
                return data_imputed
            data_columns = self.raw_data.columns
            add_columns = []
            if impute_method in self.order_func:                # If you specify a function that needs to be sorted, you need to specify which column to be sorted.
                windowspec_r = Window.orderBy(order_by).partitionBy(layer)
                add_columns = ['windowspec_r_order']            # Add a column for sorting.
                order_sp = self.raw_data.select(data_columns + [F.row_number().over(windowspec_r).alias(add_columns)])
            else:
                order_sp = self.raw_data
            func_type = self.func_type
            order_func = self.order_func
            origin_columns = list(set(data_columns) - set(target_columns))          # Gets a column name other than the target column. For forming the finished spark DataFrame.
            origin_len = len(origin_columns)
            imputed_sp = order_sp.select(layer + origin_columns + target_columns + add_columns).rdd.map(lambda x: ('|'.join(map(str, list(x)[:len(layer)])), [list(x)[len(layer):]])). \
                reduceByKey(lambda x,y: x+y).mapValues(lambda x: _imputeMethod(x, impute_method, func_type, target_columns, order_func, origin_len)). \
                flatMapValues(lambda x:x).values().toDF(origin_columns + target_columns).select(data_columns)
            return imputed_sp
        @Logger.loggerDef('impute')
        def impute(self, target_columns=[], layer=[], impute_method='', order_by=[]):
            '''
            Impute the Missing Value
            :param target_columns: The columns we want to impute.
            :param layer: Hierarchical column name
            :param impute_method: The impute method, 'first', 'last', 'mode', 'random', 'max', 'min', 'mean', 'median', 'linear', 'quadratic', 'cubic'
            :param order_by: If you specify the split_method(['linear', 'quadratic', 'cubic', 'first', 'last']), You should specify the order of the Data
            :return: sparkDataFrame
            (annotation) : 'impute_method' mean of methods:
                            (1) 'max', 'min', 'mean', 'median', 'mode' : use the statistic methods max, min, mean, median, mode(the one which number of it is most) value to impute the missing value
                            (2) 'linear', 'quadratic', 'cubic' : use linear, quadratic, cubic splines to interpolate the missing value
                            (3) 'first', 'last' : respectively use the the prior value, following value to impute the missing value
                            (4) 'random' : use random a value to impute the missing value
                            (5) If the column is numeric, it can use : 'first', 'last', 'mode', 'random', 'max', 'min', 'mean', 'median', 'linear', 'quadratic', 'cubic' methods
                                If the column is string, it can use : 'first', 'last', 'mode', 'random' methods
            '''
            target_columns, layer, impute_method, order_by = self._checkArgs(target_columns, layer, impute_method, order_by)
            imputed_sp = self._impute(layer, impute_method, target_columns, order_by)
            return imputed_sp

    spark = SparkSession.builder.appName("test").getOrCreate()          # 必须被创建在sc之前，否则会报错
    sc = SparkContext.getOrCreate()
    tt_data = zip(['0', '1', '2', '3']*25,  ['0', '1', 'Null', '3', ' ']*20, [0, 1, None, 3, 4]*20, range(100))
    sp = spark.createDataFrame(tt_data, ['sku', 'sales_str', 'sales', 'dt'])
    imputer_item = imputer(sp, key=['sku'], spark=spark)
    imputed_sp = imputer_item.impute(target_columns=['sales_str', 'sales'], layer=['sku'], impute_method='mode', order_by=['dt'])
    logger = imputer_item.logger
    transform_item = transform(imputed_sp, target_columns=['sales_str', 'sales'], trans_method='exp', logger=logger, spark=spark)
    transformed_sp = transform_item.transformData()
    transformed_sp.show()
    logger = transform_item.logger
    logger_sp = imputer_item.saveLogger()
    print logger
    logger_sp.show()


if __name__ == '__main__':
    example()


