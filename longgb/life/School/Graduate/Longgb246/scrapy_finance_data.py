#-*- coding:utf-8 -*-
'''
该脚本用于抓取财务等基本信息。
'''
import tushare as ts
import numpy as np
import pandas as pd
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
import time
import warnings
warnings.filterwarnings('ignore')


# ============================= 基本信息 =============================
read_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
save_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Finance'
log_path = r'F:\Learning\School_Master\Graduate\Codes_Data\scrapyTitDet.log'
read_files = ['name_sha.txt', 'name_szz.txt', 'name_szzx.txt', 'name_szcy.txt', 'name_shb.txt', 'name_szb.txt']

# ============================= 日志信息 =============================
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w')
logger = logging.StreamHandler()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
logger.setFormatter(formatter)
logging.getLogger('').addHandler(logger)


def printruntime(t1):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d > 0:
        print 'Run Time is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d)
    else:
        print 'Run Time is : {0} min {1:.4f} s'.format(min_d, sec_d)

def readComName(read_file):
    '''
    读取文件
    '''
    this_read = read_path + os.sep + read_file
    names = []
    for line in open(this_read):
        tmp = line.replace('\n','').split('|')
        names.append({'name':tmp[0], 'code':tmp[1]})
    return names

def stockPrice(names, start='2017-01-01', end='2017-02-23'):
    '''
    date：日期
    open：开盘价
    high：最高价
    close：收盘价
    low：最低价
    volume：成交量
    price_change：价格变动
    p_change：涨跌幅
    ma5：5日均价
    ma10：10日均价
    ma20: 20日均价
    v_ma5: 5日均量
    v_ma10: 10日均量
    v_ma20: 20日均量
    turnover: 换手率[注：指数无此项]
    '''
    stock_price = pd.DataFrame()
    for i, each in enumerate(names):
        # each = names[0]
        print each
        tmp = ts.get_hist_data(code=each['code'], start=start, end=end)
        try:
            tmp_data = tmp.reset_index()
            tmp_data['code'] = each['code']
            stock_price = pd.concat([stock_price, tmp_data])
            stock_price.index = range(len(stock_price))
        except:
            pass
    return stock_price

def saveFile(data, filename):
    '''
    保存文件
    '''
    save_path_tmp = save_path + os.sep + filename
    data.to_csv(save_path_tmp, index=False)

def mkdirFile(filename):
    '''
    创建文件夹
    '''
    if os.path.exists(filename) == False:
        os.mkdir(filename)

def getSaveStockPrice():
    '''
    获取股价的信息
    '''
    for read_file in read_files:
        # read_file = read_files[0]
        print read_file
        names = readComName(read_file)
        stock_price = stockPrice(names, start='2016-01-01', end='2017-02-23')
        saveFile(stock_price, 'stock_price_{0}.csv'.format(read_file[5:-4]))

def getBasic(func):
    '''
    基本的获取报告格式
    '''
    basic_report = pd.DataFrame()
    for x in range(1,5):
        print x
        tmp = func(2016, x)
        tmp_time = '03-31' if x == 1 else '06-30' if x == 2 else '09-30' if x == 3 else '12-31'
        tmp['report_time'] = '2016-' + tmp_time
        print " "
        basic_report = pd.concat([basic_report, tmp])
    basic_select = basic_report.sort_values(['code', 'report_time'])
    basic_select.index = range(len(basic_select))
    basic_select['code'] = basic_select['code'].astype(str)
    basic_select['name'] = basic_select['name'].astype(str)
    basic_select['report_time'] = basic_select['report_time'].astype(str)
    return basic_select

def basicReport():
    # 一、业绩报告表
    '''
    code, 代码
    name, 名称
    esp, 每股收益               -->  esp_yeji
    eps_yoy, 每股收益同比(%)      [drop]
    bvps, 每股净资产
    roe, 净资产收益率(%)        --> roe_yeji
    epcf, 每股现金流量(元)
    net_profits, 净利润(万元)   --> net_profits_yeji
    profits_yoy, 净利润同比(%)   [drop]
    distrib, 分配方案
    report_date, 发布日期
    '''
    yeji_report = getBasic(ts.get_report_data)
    yeji_report = yeji_report.drop(['eps_yoy', 'profits_yoy'], axis=1)
    yeji_report = yeji_report.rename(columns={"eps":"eps_yeji", "roe":"roe_yeji", 'net_profits':'net_profits_yeji'})
    # 9151
    # 二、盈利能力表
    '''
    code, 代码
    name, 名称
    roe, 净资产收益率(%)          --> roe_yingli
    net_profit_ratio, 净利率(%)
    gross_profit_rate, 毛利率(%)
    net_profits, 净利润(万元)    --> net_profits_yingli
    esp, 每股收益                 --> esp_yingli
    business_income, 营业收入(百万元)
    bips, 每股主营业务收入(元)
    '''
    yingli_report = getBasic(ts.get_profit_data)
    yingli_report = yingli_report.drop(['name'], axis=1)
    yingli_report = yingli_report.rename(columns={"eps": "eps_yingli", "roe": "roe_yingli", 'net_profits':'net_profits_yingli'})
    # 9233
    # 三、营运能力表
    '''
    code,代码
    name,名称
    arturnover,应收账款周转率(次)
    arturndays,应收账款周转天数(天)
    inventory_turnover,存货周转率(次)
    inventory_days,存货周转天数(天)
    currentasset_turnover,流动资产周转率(次)
    currentasset_days,流动资产周转天数(天)
    '''
    yunying_report = getBasic(ts.get_operation_data)
    yunying_report = yunying_report.drop(['name'], axis=1)
    # 9152
    # 四、成长能力表
    '''
    code,代码
    name,名称
    mbrg,主营业务收入增长率(%)
    nprg,净利润增长率(%)
    nav,净资产增长率
    targ,总资产增长率
    epsg,每股收益增长率
    seg,股东权益增长率
    '''
    chengzhang_report = getBasic(ts.get_growth_data)
    chengzhang_report = chengzhang_report.drop(['name'], axis=1)
    # 五、偿债能力表
    '''
    code,代码
    name,名称
    currentratio,流动比率
    quickratio,速动比率
    cashratio,现金比率
    icratio,利息支付倍数
    sheqratio,股东权益比率
    adratio,股东权益增长率
    '''
    changzhai_report = getBasic(ts.get_debtpaying_data)
    changzhai_report = changzhai_report.drop(['name'], axis=1)
    # 六、现金流量
    '''
    code,代码
    name,名称
    cf_sales,经营现金净流量对销售收入比率
    rateofreturn,资产的经营现金流量回报率
    cf_nm,经营现金净流量与净利润的比率
    cf_liabilities,经营现金净流量对负债比率
    cashflowratio,现金流量比率
    '''
    xianjin_report = getBasic(ts.get_cashflow_data)
    xianjin_report = xianjin_report.drop(['name'], axis=1)
    basic_report = yeji_report.merge(yingli_report, on=['code', 'report_time'], how='inner')
    basic_report = basic_report.merge(yunying_report, on=['code', 'report_time'], how='inner')
    basic_report = basic_report.merge(chengzhang_report, on=['code', 'report_time'], how='inner')
    basic_report = basic_report.merge(changzhai_report, on=['code', 'report_time'], how='inner')
    basic_report = basic_report.merge(xianjin_report, on=['code', 'report_time'], how='inner')
    return basic_report

def classReport():
    '''
    股票分类报告
    '''
    class_industry = ts.get_industry_classified()
    class_industry['code'] = class_industry['code'].astype(str)
    class_industry = class_industry.drop(['name'],axis=1)
    print " "
    class_area = ts.get_area_classified()
    class_area['code'] = class_area['code'].astype(str)
    class_area = class_area.drop(['name'], axis=1)
    return class_industry, class_area

def getReport():
    '''
    获取报告信息
    '''
    basic_report = basicReport()
    class_industry, class_area = classReport()
    basic_report = basic_report.merge(class_industry, on=['code'], how='left')
    basic_report = basic_report.merge(class_area, on=['code'], how='left')
    saveFile(basic_report, 'basic_finance_report.csv')


if __name__ == '__main__':
    mkdirFile(save_path)
    # ========================= 获取股价 =========================
    t1 = time.time()
    getSaveStockPrice()
    printruntime(t1)
    # ========================= 获取报告信息 =====================
    t1 = time.time()
    getReport()
    printruntime(t1)

