#-*- coding:utf-8 -*-
from src_test.tt_a import tt_a


def tt_b(ss=''):
    print ss
    ss2 = 'This is tt_b!'
    print ss2
    return ss + ss2


if __name__ == '__main__':
    ss = tt_a()
    tt_b(ss=ss)

