# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/7/30
  Usage   : 
"""

work_path = '/Users/longguangbin/Work/Codes/pythonstudy/longgb/Scripts/PyCode/test_basic/test_argparse'

import sys

sys.path.append(work_path)
import os

os.chdir(work_path)

import argparse
import yaml


def parse_workflow(file_name):
    with open(file_name, 'r') as f:
        wf_conf = yaml.safe_load(f)
    print('\n===========\n')
    print(wf_conf)
    print('\n===========\n')
    wf_conf.values()


parser = argparse.ArgumentParser(description='test arguments')
parser.add_argument('--conf', metavar='workflow.yaml', type=str, help='workflow2.yaml')
# parser.print_help()
config = parser.parse_args()
# print('\n===========\n')
# print(config)
conf_arg = config.conf if config.conf else 'workflow.yaml'
# print(conf_arg)
parse_workflow(conf_arg)
