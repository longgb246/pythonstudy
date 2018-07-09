#-*- coding:utf-8 -*-

def nasdaq_csv():
    """
    纳斯达克的数据
    :return:
    """
    import urllib2, csv, cookielib
    # site = "http://xueqiu.com/S/AAPL/historical.csv"
    # site= "http://www.nseindia.com/live_market/dynaContent/live_watch/get_quote/getHistoricalData.jsp?symbol=JPASSOCIAT&amp;fromDate=1-JAN-2012&amp;toDate=1-AUG-2012&amp;datePeriod=unselected&amp;hiddDwnld=true"
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}
    #req = urllib2.Request(site, headers=hdr)
    symbolTest = 'APPL'
    Exchange = 'NASDAQ'
    try:
        with open(Exchange +'.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row['Symbol'], row['Name'])
                symbol = row['Symbol'].strip()
                if '^' not in symbol:
                    site = "http://xueqiu.com/S/" + symbol + "/historical.csv"
                    req = urllib2.Request(site, headers=hdr)
                    page = urllib2.urlopen(req)
                    #content = page.read()
                    with open(Exchange + '/'+symbol+'.csv','w') as symbolCSV:
                        symbolCSV.write(page.read())
                else:
                    print 'symbol contains ^, not valid, passed...'
    except urllib2.HTTPError, e:
        print e.fp.read()


# tushare 财经分析的包
import tushare as fts


fts.get_k_data()

# 获取k线数据
# ---------
# Parameters:
# code:string       股票代码 e.g. 600848
# start:string      开始日期 format：YYYY-MM-DD 为空时取上市首日
# end:string        结束日期 format：YYYY-MM-DD 为空时取最近一个交易日
# autype:string     复权类型，qfq-前复权 hfq-后复权 None-不复权，默认为qfq
# ktype：string     数据类型，D=日k线 W=周 M=月 5=5分钟 15=15分钟 30=30分钟 60=60分钟，默认为D
# retry_count : int, 默认 3           如遇网络等问题重复执行的次数
# pause : int, 默认 0                 重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题
# return
# -------
# DataFrame
# date      交易日期 (index)
# open      开盘价
# high      最高价
# close     收盘价
# low       最低价
# volume    成交量
# amount    成交额
# turnoverratio     换手率
# code      股票代码

# fts.get_hist_data('sh'）#获取上证指数k线数据，其它参数与个股一致，下同
# fts.get_hist_data('sz'）#获取深圳成指k线数据
# fts.get_hist_data('hs300'）#获取沪深300指数k线数据
# fts.get_hist_data('sz50'）#获取上证50指数k线数据
# fts.get_hist_data('zxb'）#获取中小板指数k线数据
# fts.get_hist_data('cyb'）#获取创业板指数k线数据

# 交易数据
# # 1、分配预案，分红相关             fts.profit_data(top=60)
# # 2、业绩预告                      fts.forecast_data(2014,2)
# # 3、限售股解禁                    fts.xsg_data()
# # 4、基金持股                      fts.fund_holdings(2014, 4)
# # 5、新股数据                      fts.new_stocks()
# # 6、融资融券（沪市）               fts.sh_margins(start='2015-01-01', end='2015-04-19')

# 股票分类数据
# # 1、行业分类                      ts.get_industry_classified()
# # 2、概念分类                      ts.get_concept_classified()
# # 3、地域分类                      ts.get_area_classified()
# # 4、中小板分类                    ts.get_sme_classified()
# # 5、创业板分类                    ts.get_gem_classified()
# # 6、风险警示板分类                 ts.get_st_classified()
# # 7、沪深300成份及权重              ts.get_hs300s()
# # 8、上证50成份股                  ts.get_sz50s()
# # 9、中证500成份股                 ts.get_zz500s()
# # 10、终止上市股票列表              ts.get_terminated()
# # 11、暂停上市股票列表              ts.get_suspended()

# 基本面数据
# 股票列表          ts.get_stock_basics()
# 业绩报告（主表）   ts.get_report_data(2014,3)
# 盈利能力          ts.get_profit_data(2014,3)
# 营运能力          ts.get_operation_data(2014,3)
# 成长能力          ts.get_growth_data(2014,3)
# 偿债能力          ts.get_debtpaying_data(2014,3)
# 现金流量          ts.get_cashflow_data(2014,3)

# 宏观经济数据
# 存款利率                      ts.get_deposit_rate()
# 贷款利率                      ts.get_loan_rate()
# 存款准备金率                  ts.get_rrr()
# 货币供应量                    ts.get_money_supply()
# 货币供应量(年底余额)           ts.get_money_supply_bal()
# 国内生产总值(年度)             ts.get_gdp_year()
# 国内生产总值(季度)             ts.get_gdp_quarter()
# 三大需求对GDP贡献              ts.get_gdp_for()
# 三大产业对GDP拉动              ts.get_gdp_pull()
# 三大产业贡献率                 ts.get_gdp_contrib()
# 居民消费价格指数               ts.get_cpi()
# 工业品出厂价格指数             ts.get_ppi()

# 新闻事件数据
# 信息地雷                      ts.get_notices()            code:股票代码  date:信息公布日期
# 新浪股吧                      ts.guba_sina()


fts.realtime_boxoffice()


