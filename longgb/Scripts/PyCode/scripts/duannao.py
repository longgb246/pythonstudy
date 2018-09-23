#-*- coding:utf-8 -*-
import requests
from lxml import etree

url = 'http://manhua.mimanhua.com/cartoon/DuanNao/10662_182879_1.html'

response = requests.get(url=url)
content = etree.HTML(response.content)
content.xpath('//*[@id="pic-list"]/img')
content.xpath('//*[@id="pic-list"]')[0].attrib


a = 0
for i in range(16):
    a = a + 18 + i



print response.content
response





