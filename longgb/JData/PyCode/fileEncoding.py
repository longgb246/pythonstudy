#-*- coding:utf-8 -*-
#===============================================================================
#
#         FILE: fileEncoding.py
#
#  DESCRIPTION: 用于修改文件编码
#      OPTIONS: ---
# REQUIREMENTS: ---
#        NOTES: ---
#       AUTHOR: lgb453476610@163.com
#      COMPANY: self
#      VERSION: 1.0
#      CREATED: 2017-09-04
#===============================================================================
import os


isDelete = 1        # 是否删除第一行


if __name__ == '__main__':
    file_list = os.listdir(os.getcwd())
    for file in file_list:
        if 'csv' in file:
            print file
            if isDelete:
                os.system('''sed -i '1d' {0};'''.format(file))
            os.system('''iconv -f gbk -t utf-8 {0} -o {0};'''.format(file))

