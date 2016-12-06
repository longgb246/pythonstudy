#-*- coding: utf-8 -*-
import datetime
class DateUtil(object):


    #给定日期向后N天的日期
    @staticmethod
    def dateadd_day(days):
        d1 = datetime.datetime.now()
        d3 = d1 + datetime.timedelta(days)
        return d3
    #昨天
    @staticmethod
    def getYesterday():
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1)
        yesterday = today - oneday
        return yesterday

    #今天
    @staticmethod
    def getToday():
        return datetime.date.today()

    # 计算指定日期前后的日期,并将日期格式化
    @staticmethod
    def getDate4OneDay(datestr, date_format, days):
        d_type = DateUtil.strtodatetime(datestr, date_format)
        d = d_type + datetime.timedelta(days)
        return d.strftime(date_format)

    #获取给定参数的前几天的日期，返回一个list
    @staticmethod
    def getDaysBefore(num):
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1)
        li = []
        for i in range(0, num):
            #今天减一天，一天一天减
            today = today - oneday
            #把日期转换成字符串
            li.append(DateUtil.datetostr(today))
        return li

    #将字符串转换成datetime类型
    @staticmethod
    def strtodatetime(datestr, format):
        return datetime.datetime.strptime(datestr, format)

    #时间转换成字符串,格式为2015-02-02
    @staticmethod
    def datetostr(date):
        return str(date)[0:10]

    #时间转换成字符串,格式为2015-02-02
    @staticmethod
    def datetostr_secod(date):
        return str(date)[0:19]

    #两个日期相隔多少天，例：2015-2-04和2015-3-1
    @staticmethod
    def datediff(beginDate, endDate):
        format = "%Y-%m-%d"
        bd = DateUtil.strtodatetime(beginDate, format)
        ed = DateUtil.strtodatetime(endDate, format)
        oneday = datetime.timedelta(days=1)
        count = 0
        while bd != ed:
            ed = ed - oneday
            count += 1
        return count

    #两个日期之间相差的秒
    @staticmethod
    def datediff_seconds(beginDate, endDate):
        format = "%Y-%m-%d %H:%M:%S"
        if " " not in beginDate or ':' not in beginDate:
            bformat = "%Y-%m-%d"
        else:
            bformat = format
        if " " not in endDate or ':' not in endDate:
            eformat = "%Y-%m-%d"
        else:
            eformat = format
        starttime = DateUtil.strtodatetime(beginDate, bformat)
        endtime = DateUtil.strtodatetime(endDate, eformat)
        ret = endtime - starttime
        return ret.days * 86400 + ret.seconds

    #获取两个时间段的所有时间,返回list
    @staticmethod
    def getDays(beginDate, endDate):
        format = "%Y-%m-%d"
        begin = DateUtil.strtodatetime(beginDate, format)
        oneday = datetime.timedelta(days=1)
        num = DateUtil.datediff(beginDate, endDate) + 1
        li = []
        for i in range(0, num):
            li.append(DateUtil.datetostr(begin))
            begin = begin + oneday
        return li

    #获取当前年份 是一个字符串
    @staticmethod
    def getYear(date=datetime.date.today()):
        return str(date)[0:4]

    #获取当前月份 是一个字符串
    @staticmethod
    def getMonth(date=datetime.date.today()):
        return str(date)[5:7]

    #获取当前天 是一个字符串
    @staticmethod
    def getDay(date=datetime.date.today()):
        return str(date)[8:10]

    #获取当前小时 是一个字符串
    @staticmethod
    def getHour(date=datetime.datetime.now()):
        return str(date)[11:13]

    #获取当前分钟 是一个字符串
    def getMinute(date=datetime.datetime.now()):
        return str(date)[14:16]

    #获取当前秒 是一个字符串
    @staticmethod
    def getSecond(date=datetime.datetime.now()):
        return str(date)[17:19]

    @staticmethod
    def getNow():
        return datetime.datetime.now()

# print DateUtil.dateadd_day(10)
# #2015-02-14 16:41:13.275000
#
# print DateUtil.getYesterday()
# #2015-02-03
#
# print DateUtil.getToday()
# #2015-02-04
#
# print DateUtil.getDaysBefore(3)
# #['2015-02-03', '2015-02-02', '2015-02-01']

#print DateUtil.datediff('2016-2-01', '2016-03-01')
#246
#
# print DateUtil.datediff_seconds('2015-02-04', '2015-02-05')
# #86400
#
# print DateUtil.datediff_seconds('2015-02-04 22:00:00', '2015-02-05')
# #7200
#
# print DateUtil.getDays('2015-2-03', '2015-2-05')
# #['2015-02-03', '2015-02-04', '2015-02-05']
#
# print DateUtil.datetostr_secod(DateUtil.getNow())
# #2015-02-04 16:46:47

# print str(DateUtil.getYear(DateUtil.dateadd_day(-50))) + '-' \
#       + DateUtil.getMonth() + '-' \
#       + DateUtil.getDay() + ' ' \
#       + DateUtil.getHour() + ':' \
#       + DateUtil.getMinute() + ':' \
#       + DateUtil.getSecond()
# #2014-02-04 16:59:04

# print DateUtil.getNow()
#2015-02-04 16:46:47.454000