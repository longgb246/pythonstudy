# -*- coding:utf-8 -*-
'''
使用python27运行。
该脚本用于单线程抓取新闻标题和相应的urls。
'''
import requests
from lxml import etree
import os
from string import Template
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
import re
import chardet


class scrapyTitDet():
    def __init__(self):
        self.url_st_tem = Template('''http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=${com_name}&Page=${page}''')
        self.read_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
        self.save_path = r'F:\Learning\School_Master\Graduate\Codes_Data'
        self.log_path = r'F:\Learning\School_Master\Graduate\Codes_Data\scrapyTitDet.log'
        self.setlog()
        self.read_files = ['name_sha.txt', 'name_szz.txt', 'name_szzx.txt', 'name_szcy.txt', 'name_shb.txt', 'name_szb.txt']
        # self.read_files = ['name_sha.txt']

    def setlog(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=self.log_path,
                            filemode='w')
        self.logger = logging.StreamHandler()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        self.logger.setFormatter(formatter)
        logging.getLogger('').addHandler(self.logger)

    def mkdirs(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)

    def readComName(self):
        '''
        读取文件
        '''
        # this_read = read_path + os.sep + read_file
        this_read = self.read_path + os.sep + self.read_file
        for line in open(this_read):
            # names.append(line)
            self.names.append(line)

    def saveFile(self):
        index_name = self.mkdir_com + os.sep + 'index.txt'
        with open(index_name, 'a') as f:
            for each in self.result:
                f.write(each[0])
                f.write("  ")
                f.write(each[1])
                f.write("|")
                f.write(each[2])
                f.write("|")
                try:
                    f.write(each[3].decode('gb18030').encode('utf-8'))
                except:
                    try:
                        f.write(each[3][:-1].decode('gb18030').encode('utf-8'))
                    except:
                        print each
                        logging.error('Error Encode! ')
                        try:
                            logging.error(str(each))
                        except:
                            pass
                f.write("\n")

    def parserTitPageName(self):
        # nodes = tree.xpath('//div[@id="con02-7"]/table/tr/td/div/ul/a')
        self.result = re.findall('''&nbsp;&nbsp;&nbsp;&nbsp;(.*?)&nbsp;(.*?)&nbsp;&nbsp;<a target='_blank' href='(.*?)'>(.*?)</a> <br>''', self.r_text_utf)
        self.saveFile()

    def parserTitPage(self, page):
        # this_url = url_st_tem.substitute(page=page, com_name=read_file[5:7]+com_name[1].replace('\n',''))
        this_url = self.url_st_tem.substitute(page=page, com_name=self.read_file[5:7] + self.com_name[1].replace('\n',''))
        logging.info('[    Page In   ] : ' + str(page) + '  ' + str(this_url) + '  ...')
        r = requests.get(this_url)
        r_text_uni = r.text
        self.r_text_utf = r_text_uni.encode('raw_unicode_escape')
        self.tree = etree.HTML(self.r_text_utf)
        # tree = etree.HTML(r_text_utf)
        # nodes = tree.xpath('//div[@id="con02-7"]/table/tr/td/div/ul/a')
        self.parserTitPageName()

    def parserTit(self):
        for page in range(1, 11):
            # page = 1
            # print '[    Page In   ] : ', page , '  ...'
            self.parserTitPage(page)

    def scrapyRun(self):
        for read_file in self.read_files:
            self.read_file = read_file
            self.mkdir_file = self.save_path + os.sep + read_file.split('.')[0][5:]
            self.mkdirs(self.mkdir_file)
            # read_file = read_files[0]
            logging.info('[   Load Name  ] : '+ read_file + '  ...')
            # print '[   Load Name  ] : ', read_file, '  ...'
            self.names = []
            # ================================ 读取文件 ================================
            self.readComName()
            for com_name in self.names:
                # com_name = names[0]
                self.com_name = com_name.replace('\n', '').split('|')
                self.mkdir_com = self.mkdir_file + os.sep + self.com_name[1]
                self.mkdirs(self.mkdir_com)
                self.com_name_nodes = []
                logging.info('[ Company Name ] : '+ self.com_name[0] + '  ' + self.com_name[1] + '  ...')
                # print '[ Company Name ] : ', com_name, '  ...'
                # ============================= 解析文件 ===============================
                self.parserTit()


if __name__ == '__main__':
    sc = scrapyTitDet()
    sc.scrapyRun()



