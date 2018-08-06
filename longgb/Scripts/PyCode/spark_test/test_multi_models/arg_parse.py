#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from src.config.algorithm_config.algorithm_config import alg_conf as alg_conf
from src.config.algorithm_config.timeSeries_config import ts_conf as ts_conf
from src.config.base_config.base_config import base_conf as base_conf
from src.config.feature_config.feature_base_config import feat_base_conf as feat_base_conf
from src.config.feature_config.feature_config import feat_conf as feat_conf
from src.config.feature_config.feature_grant_config import feat_grant_conf as feat_grant_conf
from src.config.interface_config.interface_config import inter_conf as inter_conf
from src.config.param_config.param_config import param_dict as param_dict
from src.config.processor_config.pre_processor_config import pre_conf as pre_conf
from src.config.promotion_config.promotion_config import promo_conf

arg_config = {
    "alg_conf": alg_conf,
    "ts_conf": ts_conf,
    "base_conf": base_conf,
    "feat_base_conf": feat_base_conf,
    "feat_conf": feat_conf,
    "feat_grant_conf": feat_grant_conf,
    "inter_conf": inter_conf,
    "param_dict": param_dict,
    "pre_conf": pre_conf,
    "promo_conf": promo_conf
}


def pipeline_arg_parse(arg_config=arg_config):
    config = flat_config(arg_config)
    parser = argparse.ArgumentParser(description='forecast process')
    optional_prefix = '--'
    for k, v in config.items():
        parser.add_argument(optional_prefix + k, nargs='?', help=k)
    print(parser)
    args = parser.parse_args()
    for k, v in args.__dict__.items():
        v = v.replace("(#k)", ' ') if v and isinstance(v, str) else v
        _set_arg_config(k, v)
    print("parse and set config finished")
    return args


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
        elif isinstance(v, str) or isinstance(v, bool) or isinstance(v, float) or isinstance(v, int) or isinstance(v, list):
            config[k] = v
    return config


def _set_arg_config(key, value, dic=arg_config):
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



if __name__ == '__main__':
    xx = {}
    xx["Meta_Data"] = {
        "SQL_INTERFACE_DICT": "select * from dual",
        "PARAM_DICT": {
            'Tenant_id': 3,
            'id': 2
        }
    }
    c = {
        "inter_conf": xx,
    }
    conf = flat_config(xx)
    print(conf)
    x = {"a":{"b":3}}
    _set_arg_config('a.b', 4, x)
    print(x)
    print("========")
    x = {"a":3}
    _set_arg_config('a', 4, x)
    print(x)
    print("========")
    x = {"a":{"b":3,"c":66}}
    _set_arg_config('a.b', 4, x)
    print(x)