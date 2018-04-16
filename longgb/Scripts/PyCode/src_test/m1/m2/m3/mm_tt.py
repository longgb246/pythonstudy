#-*- coding:utf-8 -*-
import sys
import os
import re
def appendPath(end_dir='src'):
    '''
    Append the root path ('src') into package find path.
    '''
    this_path = re.split(r'[\\|/]', os.path.abspath(__file__))
    i = 0
    for i, each in enumerate(this_path[::-1]):
        if each == end_dir:
            break
    add_path = os.sep.join(this_path[:(-(i+1))])
    sys.path.append(add_path)
appendPath(end_dir='src_test')

from src_test.tt_a import tt_a
from src_test.tt_b import tt_b

aa = tt_a()
tt_b(aa)

