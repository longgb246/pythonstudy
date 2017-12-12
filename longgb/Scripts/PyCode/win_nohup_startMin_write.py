#-*- coding:utf-8 -*-
import sys
import time


if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = 'mm'


time.sleep(5)
with open('{0}.txt'.format(file_name), 'w') as f:
    f.writelines(' I have write some Thing!')
    f.writelines(' I have write some Thing!')
    f.writelines(' I have write some Thing!')


print 'Over  Test  Write!'

