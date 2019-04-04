#!/usr/bin/env python3
# coding:utf-8
"""
@author: "Zhangjianshen"
@contact: zhangjianshen@xxx.com

脚本使用实例
运行脚本a.py 从2016-01-01到2016-01-30，左闭右开，不包含结束日期: run_file_by_day.py a.py 2016-01-01 2016-01-30
运行脚本昨日日期  a.py: run_file_by_day.py a.py
"""
import argparse
import datetime
import re
from subprocess import call

import yaml


def get_format_yesterday(format_str='%Y-%m-%d'):
    """
    Get the yesterday using the specified format.
    :param format_str: String. A time format.
    :return: String. The string of yesterday in the specified format.
    """
    yesterday = (datetime.date.today() + datetime.timedelta(-1)).strftime(format_str)
    return yesterday


def script_arg_parse():
    parser = argparse.ArgumentParser(description='forecast process')
    parser.add_argument('script_file',
                        metavar='run_file.py zip_file.zip setting.yaml',
                        type=str, nargs='+',
                        help='script_run.py ${run_file} ${zip_file} ${setting.yaml}')
    config = parser.parse_args()
    nargs = len(config.script_file)
    run_file = config.script_file[0]
    zip_file_name = config.script_file[1] if nargs >= 2 else 'src.zip'
    setting_path = config.script_file[2] if nargs >= 3 else 'setting.yaml'
    return run_file, zip_file_name, setting_path


def parse_command(setting_file_name):
    with open(setting_file_name, 'r') as setting_file:
        setting = yaml.safe_load(setting_file)
        return setting['spark_options'], setting['spark_conf'], setting['algo_conf']


def parse_args_time(algo_conf):
    """
    将时间转换成datetime格式
    :param algo_conf:
    :return:
    """
    today = datetime.date.today()
    sfs_dt = algo_conf['param_dict.sfs_dt'] if 'param_dict.sfs_dt' in algo_conf and algo_conf[
        'param_dict.sfs_dt'] else today + datetime.timedelta(-1)
    end_date = algo_conf['param_dict.end_date'] if 'param_dict.end_date' in algo_conf and algo_conf[
        'param_dict.end_date'] else today
    return sfs_dt, end_date


def format_args_days(algo_conf, sfs_dt, end_date):
    """
    将时间的格式format成字符串
    在参数前加--用于解析
    part_date
    :param algo_conf:
    :param sfs_dt: 读取的分区
    :param end_date: 训练的终止时间
    :return: str
    """
    algo_conf['param_dict.sfs_dt'] = sfs_dt.strftime('%Y-%m-%d')
    algo_conf['param_dict.end_date'] = end_date.strftime('%Y-%m-%d')
    algo_conf['param_dict.part_date'] = sfs_dt.strftime('%Y-%m-%d')
    return ' '.join([' --' + str(k) + ' ' + str(v) for (k, v) in algo_conf.items()])


def concat_command(spark_options, spark_conf, algo_conf):
    command = 'spark-submit'
    command += ''.join([' --' + str(k) + ' ' + str(v) for (k, v) in spark_options.items()])
    command += ''.join([' --conf ' + str(k) + '=' + str(v) for (k, v) in spark_conf.items()])
    command += ' --py-files {zip_files} {pyfile} {args}'
    for k, v in algo_conf.items():
        if isinstance(v, str):
            algo_conf[k] = v.replace(' ', '(#k)')
    return command


# python script_run.py run_feature.py src.zip daylevel_setting.yaml data_split=3
if __name__ == '__main__':
    pyfile, zip_files, setting_file_name = script_arg_parse()
    spark_options, spark_conf, algo_conf = parse_command(setting_file_name)
    calc_day, finish_date = parse_args_time(algo_conf)
    while calc_day != finish_date:
        command_template = concat_command(spark_options, spark_conf, algo_conf)
        spark_conf_args = format_args_days(algo_conf, calc_day, finish_date)
        print ''
        print 'command_template : ', command_template
        print 'pyfile : ', pyfile
        print 'zip_files : ', zip_files
        print 'spark_conf_args : ', spark_conf_args
        print ''
        exit()
        command = command_template.format(pyfile=pyfile, zip_files=zip_files, args=spark_conf_args)
        print(command)
        cmd = re.split('\s+', command)
        ok = call(cmd)
        print('\n' + 'return status：' + str(ok) + '\n')
        if ok == 0:
            print('----------------------- Successfully -----------------------')
        else:
            print('--------------------------- Error --------------------------')
            raise Exception('ERROR：' + str(ok))
        calc_day = calc_day + datetime.timedelta(1)
    exit(0)
