#!/usr/bin/env python3
# coding:utf-8

import os
import datetime
import sys


def get_format_yesterday(format='%Y-%m-%d'):
    yesterday = (datetime.date.today() + datetime.timedelta(-1)).strftime(format)
    return yesterday

"""
脚本使用实例
运行脚本a.py 从2016-01-01到2016-01-30，左闭右开，不包含结束日期: run_file_by_day.py a.py 2016-01-01 2016-01-30
运行脚本昨日日期  a.py: run_file_by_day.py a.py
"""
if __name__ == '__main__':
    today = datetime.date.today()
    if (len(sys.argv) > 3):
        if (sys.argv[2] < sys.argv[3]):
            pyFile = sys.argv[1]
            starDate = datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
            finishDate = datetime.datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
        else:
            raise RuntimeError("end time must bigger than start time")
    else:
        pyFile = sys.argv[1]
        starDate = today + datetime.timedelta(-1)
        finishDate = today

    strDay = ''
    calcDay = starDate

    while (calcDay != finishDate):
        strDay = calcDay.strftime('%Y-%m-%d')
        print("python3 " + pyFile + " " + strDay)
        os.system("""python3 """ + pyFile + """ """ + strDay)
        calcDay = calcDay + datetime.timedelta(1)
