#-*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/3
  Usage   : 
"""

import argparse


def script_arg_parse():
    parser = argparse.ArgumentParser(description='forecast process')
    parser.add_argument('script_file',
                        metavar='run_file.py zip_file.zip setting.yaml',
                        type=str, nargs='+',
                        help='script_run.py ${run_file} ${zip_file} ${setting.yaml}')
                        # help='script_run.py %(prog)s')
    # nargs = len(config.script_file)
    # run_file = config.script_file[0]
    # zip_file_name = config.script_file[1] if nargs >= 2 else 'src.zip'
    # setting_path = config.script_file[2] if nargs >= 3 else 'setting.yaml'

    parser.add_argument('--infile.aa', nargs='?', help='aaa')
    parser.add_argument('--infile.bb', nargs='?', help='aaa')

    config = parser.parse_args()

    # other_conf = config.script_file[3:]

    print ''
    # parser.print_help()
    print config.__dict__
    # print 'run_file : ', run_file
    # print 'zip_file_name : ', zip_file_name
    # print 'setting_path : ', setting_path
    # print 'other_conf : ', other_conf
    print ''

    # return run_file, zip_file_name, setting_path


# python test_args_parse.py component/run_feature.py src.zip component/daylevel_setting.yaml --help=1
# python test_args_parse.py --infile.aa cc --infile.bb dd
script_arg_parse()




