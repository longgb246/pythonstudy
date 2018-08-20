#-*- coding:utf-8 -*-
from __future__ import print_function
import time


for i in range(10):
    for j in range(10):
        print('[================================] ( {0}/{1} )'.format(j+1, 10))
    time.sleep(1)
    print('\x1b[2J')
    print('\x1b[1A\x1b[2K'+'\r*****  '+'\x1b[1;31m'+'Not A Number!'+'\x1b[0m'+'  *****', end='')




