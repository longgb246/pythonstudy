#!/usr/bin/env python
#coding:utf-8
#Author:liushsh

import sys
import logging
import logging.config
import re
import cookielib
import urllib,urllib2
import random
import json

# logging.config.fileConfig("conf/logger.conf")
# logger = logging.getLogger('root')

def Translation(en_str,zh_str):
    # set cookie process
    cj = cookielib.CookieJar()
    cookie_support = urllib2.HTTPCookieProcessor(cj)
    opener = urllib2.build_opener(cookie_support, urllib2.HTTPHandler)
    urllib2.install_opener(opener)

    # simulation browse load host url,get cookie
    # http://fanyi.baidu.com/v2transapi?from=en&to=zh&query=hello%20world&transtype=hash
    baidu_translation='''''http://fanyi.baidu.com/v2transapi'''
    post_param={'from':'en','to':'zh','query':en_str,'transtype':'hash'}
    post_data=urllib.urlencode(post_param)

    hostUrl = baidu_translation
    domain = urllib2.Request(hostUrl,post_data)
    f = urllib2.urlopen(domain)
    html = f.read()
    #logger.info("json:[%s]"%(html))

    j = json.loads(html)
    if j['trans_result']['data'][0]['dst'] != None:
        zh_str[0] = j['trans_result']['data'][0]['dst']
        # logger.info("[%s]==>[%s]"%(en_str,zh_str[0]))
        print "[%s]==>[%s]"%(en_str,zh_str[0])
        return True
    return False

def main():
    argc=len(sys.argv)
    if argc < 2:
        print "Usage:%s 'hello world'"%(sys.argv[0])
        exit(0)

    for en_str in sys.argv[1:len(sys.argv)]:
        try:
            zh_str = [""]
            Translation(en_str,zh_str)
            print "[%s]==>[%s]"%(en_str,zh_str[0])
        except urllib2.URLError,e:
            print "[except:%s]"%(e)
        except Exception,e1:
            print "[except:%s]"%(e1)

if __name__ == "__main__":
    main()