# -*- coding:utf-8 -*-
import httplib
import md5
import urllib
import random
import chardet
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json

appid = '20170222000039667'                     # 账号
secretKey = 'IeZ2wtwnNYxszXPDYDzC'              # 密钥

httpClient = None
myurl = '/api/trans/vip/translate'
q = 'apple'                                     # 翻译语句
fromLang = 'en'                                 # 从..语言翻译
toLang = 'zh'                                   # 翻译到..语言
salt = random.randint(32768, 65536)             # 随机数

sign = appid + q + str(salt) + secretKey        # sign
m1 = md5.new()
m1.update(sign)
sign = m1.hexdigest()                           # md5
myurl = myurl + '?appid=' + appid + '&q=' + urllib.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

try:
    httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)
    # response是HTTPResponse对象
    response = httpClient.getresponse()
    # print response.read()
    response_dict2 = response.read()
    response_dict2 = json.loads(response_dict2)
    print response_dict2["trans_result"][0]["dst"]
    response_dict = eval(response.read())
    # print response_dict["trans_result"][0]["dst"]
    result = response_dict["trans_result"][0]["dst"]
    # print chardet.detect(result)
    uni_result = result.decode("unicode-escape")
    print uni_result
except Exception, e:
    print e
finally:
    if httpClient:
        httpClient.close()
