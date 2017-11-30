#-*- coding:utf-8 -*-
import multiprocessing
import time
import sys

def daemon():
    name = multiprocessing.current_process().name
    print 'Starting:', name
    time.sleep(5)
    print 'Exiting :', name

def non_daemon():
    name = multiprocessing.current_process().name
    print 'Starting:', name
    time.sleep(5)
    print 'Exiting :', name

if __name__ == '__main__':
    d = multiprocessing.Process(name='daemon',
                                target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon',
                                target=non_daemon)
    n.daemon = False

    d.start()
    n.start()

    print 'd.is_alive()', d.is_alive()
    print 'n.is_alive()', n.is_alive()
    time.sleep(1)
    exit()
    print 'd.is_alive()', d.is_alive()
    print 'n.is_alive()', n.is_alive()
