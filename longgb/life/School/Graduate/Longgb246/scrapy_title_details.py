# -*- coding:utf-8 -*-
'''
该脚本用于
'''
import requests
from lxml import etree
import os
from string import Template
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class scrapyTitDet():
    def __init__(self):
        self.url_st_tem = Template('''http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=${code}&Page=${page}''')
        self.read_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
        self.save_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
        self.read_files = ['name_sha.txt', 'name_szz.txt', 'name_szzx.txt', 'name_szcy.txt', 'name_shb.txt', 'name_szb.txt']

    def getComName(self):
        this_read = self.read_files[0]


if __name__ == '__main__':
    pass


# #con02-7 > table > tbody > tr:nth-child(2) > td > div.datelist > ul
# //*[@id="con02-7"]/table/tbody/tr[2]/td/div[1]/ul

