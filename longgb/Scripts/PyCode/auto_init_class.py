#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: zhangjianshen
@file: auto_init_class
@time: 2018/10/22
'''
import inspect


# f_args = inspect.getargspec(barplot)
# dict(f_args._asdict())

def fn_auto_init_class(auto_class, input_dict):
    assert isinstance(input_dict, dict), ValueError('input_dict must be a dict')
    f = auto_class.__init__
    f_args = inspect.getargspec(f)
    if f_args.defaults is not None:
        f_args_tuple = list(zip(f_args.args[-len(f_args.defaults):], f_args.defaults))
        f_args_tuple.extend(
            list(zip(f_args.args[1:-len(f_args.defaults)], [None] * int(len(f_args.args[1:-len(f_args.defaults)])))))
    else:
        f_args_tuple = list(zip(f_args.args[1:], [None] * int(len(f_args.args[1:]))))
    f_args_dict = dict(f_args_tuple)
    f_args_dict_key = f_args_dict.keys()
    mid_input_dict = {key: input_dict.get(key, f_args_dict.get(key)) for key in f_args_dict_key}
    f_args_dict.update(mid_input_dict)
    return auto_class(**f_args_dict)
