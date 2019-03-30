# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/3/25
"""  
Usage Of 'mm.py' : 
"""


def fn(self, name="world"):
    print("Hello,%s" % name)


Hello = type('Hello', (object,), dict(hello=fn))
h = Hello()
h.hello()
print(type(Hello))


class FillTime(object):
    def __init__(self, data, conf={}):
        print data, conf

    def _fill_lost_value(self):
        print('just test')


class FillTime2(object):
    def __init__(self, data, conf={}):
        print data, conf, '2'

    def _fill_lost_value(self):
        print('just test 2')


def gene_class_from_class(class_name):
    def execute(self, data, conf={}, **kwargs):
        conf = dict(conf, **kwargs)
        ft = globals().get(class_name)(data, **conf)
        ft._fill_lost_value()

    new_class_name = str(class_name) + 'Executor'
    return type(new_class_name, (object,), dict(execute=execute))


mm = gene_class_from_class('FillTime')
print mm
mm().execute('1')

mm = gene_class_from_class('FillTime2')
print mm
mm().execute('2')
