# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/3/16
"""  
Usage Of 'new_luwen.py' : 
"""

from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import re
import datetime
import requests
from dateutil.parser import parse
import pandas as pd
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')


def get_urls(file_path, url_save_path, root_url):
    with open(file_path, 'r') as f:
        all_html = f.read()

    html_list = re.findall(ur'(<html.*?</html>)', all_html, re.S)

    res = []

    for each_html in html_list:
        soup = BeautifulSoup(each_html)
        div_list = soup.find_all(name='div', attrs={"class": "row"})
        tmp = []
        for div in div_list:
            div.find_all('li')
            index = div.find('li', attrs={"class": "xh"}).text  # 序号
            year = parse('-'.join(re.findall(r'(\d+)', div.find('li', attrs={"class": "fbrq"}).text))). \
                strftime('%Y-%m-%d')  # 年月
            url = root_url + div.find('li', attrs={"class": "mc"}).find('a').get('href').split('?keywords')[0]
            tmp.append([index, year, url])
        res.extend(tmp)

    url_pd = pd.DataFrame(res, columns=['index', 'year', 'url'])
    url_pd.to_csv(url_save_path, index=False, encoding='utf-8')


def find_type(s):
    tmp = re.findall(ur'[\(|\uff08](.*?)[\)|\uff09]', s)
    if len(tmp) > 0:
        return [s, tmp[-1]]
    else:
        return []


def filter_name(s):
    if s.startswith(u'(') or s.startswith(u'（'):
        try:
            index = s.index(u'）') + 1
        except:
            try:
                index = s.index(u')') + 1
            except:
                index = 0
        s = s[index:]
    s = s.replace(u'。', ''). \
        replace(u',', '').replace(u'，', ''). \
        replace(u';', '').replace(u'；', '')
    return s


def get_pass(s):
    try:
        index = s.rindex(u'）') + 1
    except:
        try:
            index = s.rindex(u')') + 1
        except:
            index = 0
    s = s[index:]
    return s


def get_company(s):
    try:
        index = s.rindex(u'（')
    except:
        try:
            index = s.rindex(u'(')
        except:
            index = 0
    index = max(0, index)
    s = s[:index]
    return s


def read_infos(url_read_path, save_company_path):
    # url_read_path = r'/Users/longguangbin/tmp/yuhan/org_data/urls.csv'
    url_pd = pd.read_csv(url_read_path, encoding='utf-8')
    must_words = [u'公司']
    or_words = [u'通过', u'暂缓']
    res_list = []
    for i in range(len(url_pd)):
        if (divmod(i, 50)[1] == 0) or (i == len(url_pd) - 1):
            print('Round : ', i)
        this_row = url_pd.loc[i, :]
        year = this_row['year']
        this_url = this_row['url']
        res = requests.get(this_url)
        new_soup = BeautifulSoup(res.content)
        contents = new_soup.find('div', attrs={"class": 'mainContainer'}).text
        contents2 = [x for x in contents.split() if all(map(lambda y: y in x, must_words))]
        contents2 = [x for x in contents2 if any(map(lambda y: y in x, or_words))]
        contents_list = [y + [year] for y in map(lambda x: find_type(x), contents2) if len(y) > 0]
        res_list.extend(contents_list)
    res_pd = pd.DataFrame(res_list, columns=['contents', 'type', 'year'])
    # res_pd.to_csv(r'/Users/longguangbin/tmp/yuhan/org_data/companys2.csv', index=False, encoding='gbk')
    res_pd.to_csv(save_company_path, index=False, encoding='gbk')


def data_filter(read_company_path, save_result_path):
    # read_company_path = r'/Users/longguangbin/tmp/yuhan/org_data/companys.csv'
    res_pd = pd.read_csv(read_company_path, encoding='gbk')

    # res_pd['contents'].apply(len).drop_duplicates().sort_values()
    cond = res_pd['contents'].apply(lambda x: len(x) < 50)
    res_pd = res_pd.loc[cond, :]

    res_pd['contents'] = res_pd['contents'].apply(filter_name)
    res_pd['company'] = res_pd['contents'].apply(get_company)
    res_pd['type'] = res_pd['contents'].apply(lambda x: find_type(x)[1])
    res_pd['pass'] = res_pd['contents'].apply(get_pass)

    res_pd.to_csv(save_result_path, index=False, encoding='gbk')


def find_col(v, cols):
    col = ''
    # cols = resset_pd.columns
    # v = u'CompanyCode'
    for each in cols:
        if v in each:
            col = each
            break
    return col


def arrange_data(resset_path):
    resset_files = [x for x in os.listdir(resset_path) if re.match(r'FININD.*?xls', x)]

    resset_data = []
    for each in resset_files:
        print('Read the File : ', each)
        # each = resset_files[0]
        tmp_pd = pd.read_excel(resset_path + os.sep + each)
        resset_data.append(tmp_pd)

    resset_pd = pd.concat(resset_data)
    resset_pd = resset_pd.drop_duplicates()
    resset_pd.to_excel(resset_path + os.sep + 'financial_all_data.xlsx', index=False, encoding='gbk')


def filter_columns(resset_path):
    resset_pd = pd.read_excel(resset_path + os.sep + 'financial_all_data.xlsx')

    cols = ['Comcd', 'Lcomnm', 'Reporttype', 'EndDt', 'Incmope', 'Netprf', 'Totcurass',
            'Totass', 'Totcurlia', 'Totlia', 'Totalprf']

    re_cols = map(lambda x: find_col(x, resset_pd.columns), cols)

    # 必须为调整之后的数据，更准确
    cond1 = resset_pd[find_col('Adjflg', resset_pd.columns)].astype(str) == '1'

    # 仅仅截取 Q4数据
    cond2 = resset_pd[find_col('Reporttype', resset_pd.columns)].astype(str) == 'Q4'

    # 截取年份
    resset_pd['year'] = resset_pd[find_col('EndDt', resset_pd.columns)]. \
        astype(str).apply(lambda x: parse(x).strftime('%Y'))

    keep_pd = resset_pd.loc[cond1 & cond2, re_cols + ['year']].drop_duplicates(). \
        sort_values([find_col('Comcd', resset_pd.columns), 'year'])
    keep_pd.to_excel(resset_path + os.sep + 'financial_keep_data.xlsx', index=False, encoding='gbk')


def to_module_data(resset_path):
    financial_keep_data = pd.read_excel(resset_path + os.sep + 'financial_keep_data.xlsx')
    financial_keep_cols = map(lambda x: x.split('_')[-1], financial_keep_data.columns)
    financial_keep_data.columns = financial_keep_cols
    financial_keep_data['Comcd'] = financial_keep_data['Comcd'].apply(lambda x: x[1:] if len(x) > 6 else x)
    financial_keep_data['Comcd'] = financial_keep_data['Comcd'].astype(str)
    financial_keep_data.to_csv(resset_path + os.sep + 'control_vars.csv', index=False, encoding='gbk')

    country_data = pd.read_excel(resset_path + os.sep + 'country_data.xlsx', converters={'code': str})
    country_data_cols = ['year', 'code', 'soe', 'sti']
    country_data.columns = country_data_cols
    country_data = country_data.drop_duplicates()
    country_data.to_csv(resset_path + os.sep + 'control_vars2.csv', index=False)


def main():
    root_url = 'http://www.csrc.gov.cn'

    file_path = r'/Users/longguangbin/tmp/yuhan/org_data/contents.txt'
    url_save_path = r'/Users/longguangbin/tmp/yuhan/org_data/urls.csv'
    save_company_path = r'/Users/longguangbin/tmp/yuhan/org_data/companys.csv'
    save_result_path = r'/Users/longguangbin/tmp/yuhan/org_data/results.csv'

    # # 爬虫爬取
    # get_urls(file_path, url_save_path, root_url)

    # # url 解析
    # read_infos(url_save_path, save_company_path)

    # # 不知道在干嘛
    # data_filter(save_company_path, save_result_path)

    # -------------------------------
    # 锐思数据库 - www.resset.cn
    resset_path = r'/Users/longguangbin/tmp/yuhan/financial_data'

    arrange_data(resset_path)
    filter_columns(resset_path)
    to_module_data(resset_path)


if __name__ == '__main__':
    main()


# -------------------------------
# 测试 pdf 解析
def pdf_parse():
    import pdfplumber

    # new_pdf = 'http://static.cninfo.com.cn/finalpage/2018-04-24/1204697395.PDF'

    new_pdf = '/Users/longguangbin/tmp/yuhan/year_report/1204697395.PDF'

    with pdfplumber.open(new_pdf) as pdf:
        first_page = pdf.pages[0]
        print(first_page.chars[0])

    dir(first_page)
    print(first_page.extract_text())

    menu = pdf.pages[2].extract_text().split('\n')
    table_start_index = 0
    for each in menu:
        if u'董事' in each:
            table_start_index = int(each.split()[-1])
            break

    print(pdf.pages[50].extract_text())
    dir(pdf.pages[50])

    table1 = pdf.pages[50].extract_tables()

    len(table1[0])

    for each in table1[0][0]:
        print(each)

    len(table1)

    for each in table1[0][4]:
        print(each)


# -------------------------------
# 测试 tushare 数据 - 失望
import tushare as ts

ts.get_stock_basics()
