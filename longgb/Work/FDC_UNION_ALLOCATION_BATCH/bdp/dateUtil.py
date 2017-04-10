# coding=utf-8


import datetime
import time
import calendar

def get_format_today(format = '%Y-%m-%d'):
    """
    # 获取今天日期（字符串） 默认'%Y-%m-%d'
    """
    today = datetime.date.today().strftime(format)
    return today

def get_format_yesterday(format = '%Y-%m-%d'):
    """
    获取昨天日期（字符串）默认'%Y-%m-%d'，
    format =‘%d' 获取昨天是本月中第几天
    """
    yesterday = (datetime.date.today() + datetime.timedelta(-1)).strftime(format)
    return yesterday

def get_current_month_begin(tx_data_str = get_format_yesterday(),format = '%Y-%m-%d'):
    """
    获取本月第一天（字符串）,默认获取昨天所在月份的第一天
    """
    tx_data = datetime.datetime.fromtimestamp(time.mktime(time.strptime(tx_data_str, format)))
    pre_month_end = datetime.date(tx_data.year, tx_data.month, 1)
    return pre_month_end.strftime(format)

def get_current_month_end(tx_data_str = get_format_yesterday(),format = '%Y-%m-%d'):
    """
    获取本月最后（字符串）,默认获取昨天所在月份的最后一天
    """
    tx_data = datetime.datetime.fromtimestamp(time.mktime(time.strptime(tx_data_str, format)))
    pre_month_end = datetime.date(tx_data.year, tx_data.month, calendar.monthrange(tx_data.year,tx_data.month)[1])
    return pre_month_end.strftime(format)

def get_week_begin(day_str = get_format_yesterday(), format = '%Y-%m-%d',N = 0):
    """
    获取指定天 所在周周一
    """
    calc = datetime.datetime.strptime(day_str,format)
    monday = calc + datetime.timedelta(-calc.weekday())
    return monday.strftime(format)

def get_week_end(day_str = get_format_yesterday(), format = '%Y-%m-%d',N = 0):
    """
    获取指定天 所在周周日
    """
    calc = datetime.datetime.strptime(day_str,format)
    monday = calc + datetime.timedelta(6-calc.weekday())
    return monday.strftime(format)

def oneday_by_date(tx_date_str, off_size, format):
    """
    获取第N天（字符串）
    """
    tx_date = datetime.datetime.fromtimestamp(time.mktime(time.strptime(tx_date_str, format)))
    yesterday = (tx_date + datetime.timedelta(off_size)).strftime(format)
    return yesterday

def get_data_days_diff(time_start,time_end,format):
    """
    获取日期相隔天数
    """
    dateStart = time.strptime(time_start,format)
    dateEnd = time.strptime(time_end,format)
    time1 = time.mktime(dateEnd)
    time2 = time.mktime(dateStart)
    daysec = 24 * 60 * 60
    return int(( time2 - time1 )/daysec)

def get_month_days(tx_data_str = get_format_yesterday(),format = '%Y-%m-%d'):
    """
        获取该月天数
    """
    tx_data = datetime.datetime.fromtimestamp(time.mktime(time.strptime(tx_data_str, format)))
    return calendar.monthrange(tx_data.year,tx_data.month)[1]

def main():
    print("get_format_today():" + get_format_today())
    print("get_format_yesterday():" + get_format_yesterday())
    print("get_format_yesterday('%d'):" + get_format_yesterday('%d'))
    print("get_current_month_begin():" + get_current_month_begin())
    print("get_current_month_end():" + get_current_month_end())
    print(get_month_days())
    print("get_week_begin:" + get_week_begin('2016-03-21'))
    print("get_week_end:" + get_week_end('2016-03-21'))

if __name__ == "__main__":
    main()