# -*- coding:utf-8 -*-
'''
该脚本用于抓取公司名称和代码。
存储于"F:\Learning\School_Master\Graduate\Codes_Data"下,命名为：
"name_sha.txt":上海A股
"name_szz.txt":深圳主板
"name_szzx.txt":深圳中小板
"name_szcy.txt":深圳创业板
"name_shb.txt":上海B股
"name_szb.txt":深圳B股
'''
# [u'上海A股', u'深圳主板', u'深圳中小板', u'深圳创业板', u'上海B股', u'深圳B股']
# ['name_sha.txt', 'name_szz.txt', 'name_szzx.txt', 'name_szcy.txt', 'name_shb.txt', 'name_szb.txt']
import requests
from lxml import etree
from bs4 import BeautifulSoup
import chardet
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class scrapyComName():
    def __init__(self):
        # 设置初始urls
        self.name_url_list = [{'url': 'http://quote.cfi.cn/stockList.aspx?t={0}'.format(url), 'url_name': url_name} for
                              url, url_name in zip(range(11, 17), ['name_sha.txt', 'name_szz.txt', 'name_szzx.txt', 'name_szcy.txt', 'name_shb.txt', 'name_szb.txt'])]
        self.save_path = r'F:\Learning\School_Master\Graduate\Codes_Data'

    def startRun(self):
        for each in self.name_url_list:
            print 'ScrapyIng...  \'',each['url'], '\'  ', each['url_name']
            self.nodes_str = []
            this_save_path = self.save_path + os.sep + each['url_name']
            self.scrapyComName(each['url'])
            self.saveComName(this_save_path)

    def scrapyComName(self, name_url):
        r = requests.get(name_url)
        # print r.text
        r_text_uni = r.text
        r_text_utf = r_text_uni.encode('utf-8')
        self.tree = etree.HTML(r_text_utf)
        self.parserComName()

    def parserComName(self):
        nodes = self.tree.xpath('body/div/table/tr/td/div/table/tr/td/a')
        for each in nodes:
            tmp = each.text.encode('raw_unicode_escape')
            result = tmp[:-8] + '|' + tmp[-7:-1]
            self.nodes_str.append(result)

    def saveComName(self, this_save_path):
        with open(this_save_path, 'w') as f:
            for each in self.nodes_str:
                f.write(each)
                f.write('\n')


if __name__ == '__main__':
    sc = scrapyComName()
    sc.startRun()
