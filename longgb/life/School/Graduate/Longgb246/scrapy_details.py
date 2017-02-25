# -*- coding:utf-8 -*-
'''
使用python27运行。
该脚本用于取新闻的内容。
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import requests
from bs4 import BeautifulSoup
import chardet
import warnings
warnings.filterwarnings('ignore')
import logging
import time


# ============================= 基本信息 =============================
read_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
read_files = ['sha', 'shb', 'szb', 'szcy', 'szz', 'szzx']
log_path = r'F:\Learning\School_Master\Graduate\Codes_Data\scrapyTitDet.log'


# ============================= 日志信息 =============================
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w')
logger = logging.StreamHandler()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
logger.setFormatter(formatter)
logging.getLogger('').addHandler(logger)

# 用于解决问题：
# HTTPConnectionPool(host='dds.cr.usgs.gov', port=80): Max retries exceeded with url: /ltaauth//sno18/ops/l1/2016/138/037/LC81380372016038LGN00.tar.gz?id=stfb9e0bgrpmc4j9lcg45ikrj1&iid=LC81380372016038LGN00&did=227966479&ver=production (Caused by NewConnectionError('<requests.packages.urllib3.connection.HTTPConnection object at 0x105b9d210>: Failed to establish a new connection: [Errno 65] No route to host',))
# 1、增加重试连接次数
requests.adapters.DEFAULT_RETRIES = 5
# DEFAULT_OPEN_PAGE_FREQUENCY = 1
# 2、关闭多余的连接
s = requests.session()
s.keep_alive = False


def parserUrl(url):
    '''
    用于解析url
    '''
    # url = 'http://finance.sina.com.cn/stock/s/2017-02-23/doc-ifyavwcv8658406.shtml'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text)
    item = soup.find_all('p')
    need_sents = []
    # print len(item)
    for each in item[:-3]:
        # each = item[0]
        mm = each.get_text()
        mm2 = mm.encode('raw_unicode_escape')
        # print chardet.detect(mm2)
        try:
            mm2 = mm2.decode('gb18030')
            # print mm2
            # print 1, u'新浪' in mm2
            if u'新浪' in mm2:
                pass
            else:
                mm2 = mm2.replace("\n", "")
                need_sents.append(mm2)
                # print mm2
        except:
            try:
                mm2 = mm2.decode('utf-8')
                # print 2, u'\u65b0\u6d6a' in mm2
                # if '新浪' in mm2:
                if u'\u65b0\u6d6a' in mm2:
                    pass
                else:
                    mm2 = mm2.replace("\n", "")
                    need_sents.append(mm2)
                    # print mm2
            except:
                pass
    try:
        result = reduce(lambda x, y: x+y, need_sents)
    except:
        result = ''
    result =  result.replace("\n", "").replace(" ", "")
    # print result
    # exit()
    return result


def loadUrls(read_file, f_i):
    '''
    用于读取url
    '''
    # read_file = read_files[0]
    root_path = read_path + os.sep + read_file
    root_files = os.listdir(root_path)
    if f_i == 0:
        flag = 0
    else:
        flag = 1
    for file in root_files:
        logging.warning('[ File ]' + file)
        if file == '600251':
            flag = 1
        if flag == 1:
            logging.warning('Run ...')
            current_path = os.path.join(root_path, file)
            load_file = current_path + os.sep + 'index.txt'
            load_urls = []
            for line in open(load_file):
                load_urls.append(line)
            # print current_path
            with open(current_path + os.sep + 'content.txt', 'w') as f:
                for line in load_urls:
                    data = line.replace('\n','').split('|')
                    logging.warning('[ News ]' + data[0] + ' | ' + data[1])
                    content = parserUrl(data[1])
                    # time.sleep(DEFAULT_OPEN_PAGE_FREQUENCY)
                    f.write(line.replace('\n',''))
                    f.write("|")
                    f.write(content.encode('utf-8'))
                    f.write('\n')

def runGetContent():
    for i, read_file in enumerate(read_files):
        logging.warning('[ Root File ]' + read_file)
        loadUrls(read_file, i)


if __name__ == '__main__':
    runGetContent()
