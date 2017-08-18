#-*- coding:utf-8 -*-
import os


if __name__ == '__main__':
    path = os.getcwd()
    sku_id = '10001'
    file = path + os.sep + 'test.txt'
    f = open(file, 'w')
    for i in range(1, 10):
        f.write(sku_id)
        f.write('\t')
        f.write('2017-01-0{0}'.format(i))
        f.write('\t')
        f.write('{0}'.format(i*10))
        f.write('\n')
    f.close()

