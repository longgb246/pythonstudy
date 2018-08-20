#-*- coding:utf-8 -*-
import sys
from time import sleep


print "\rHello, Gay! ",  '5'

for i in range(1,5):
    print "\rHello, Gay! ",  i,
    sys.stdout.flush()
    sleep(1)

print "\rHello, Gay! ",  '6'

