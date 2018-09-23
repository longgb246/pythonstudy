# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/9/6
  Usage   : 
"""

import yaml

with open('daylevel_setting.yaml', 'r') as f:
    conf = yaml.safe_load(f)
print(conf)
