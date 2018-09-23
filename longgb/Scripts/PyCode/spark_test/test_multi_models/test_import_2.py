# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/8/4
  Usage   : 
"""

import argparse

from test_import_1 import conf_tt


conf_a = {
    'conf_tt': conf_tt
}


def change_conf(conf=conf_a):
    config = flat_config(conf)
    parser = argparse.ArgumentParser(description='forecast process')
    # print(config)
    optional_prefix = '--'
    for k, v in config.items():
        parser.add_argument(optional_prefix + k, nargs='?', help=k)
    args = parser.parse_args()
    for k, v in args.__dict__.items():
        v = v.replace("(#k)", ' ') if v and isinstance(v, str) else v
        _set_arg_config(k, v)
    # print(conf_a)
    print("parse and set config finished")


def flat_config(config_item):
    '''
    transform the nest config to a flat format, such as {a: {b:1}} to a.b = 1
    :param arg_config:
    :return:
    '''
    config = {}
    for k, v in config_item.items():
        prefix = str(k) + '.'
        if isinstance(v, dict):
            flated = {prefix + kk: val for (kk, val) in flat_config(v).items()}
            config = dict(config, **flated)
        elif isinstance(v, str) or isinstance(v, bool) or isinstance(v, float) or isinstance(v, int) or \
                isinstance(v, list):
            config[k] = v
    return config


def _set_arg_config(key, value, dic=conf_a):
    if not value:
        return
    args = key.split(".")
    if len(args) < 1:
        return
    pa = dic
    key = args[-1]
    for arg_key in args[:-1]:
        pa = pa[arg_key]
    if isinstance(pa[key], bool):
        pa[key] = value == str(True)
    elif isinstance(pa[key], float):
        pa[key] = float(value)
    elif isinstance(pa[key], int):
        pa[key] = int(value)
    elif isinstance(pa[key], list):
        if value:
            pa[key] = list(value.split(","))
    else:
        pa[key] = value

