#-*- coding:utf-8 -*-
import os
import pandas as pd
from threading import Thread
from Queue import Queue
import time
import requests


def mkdir(mkdir_path):
    '''
    Make dir.
    '''
    if not os.path.exists(mkdir_path):
        os.mkdir(mkdir_path)


def loadStyleIdentifyData(read_path, columns):
    '''
    Load the StyleIdentify data.
    '''
    with open(read_path, 'r') as f:
        contents = f.readlines()
    style_pd = pd.DataFrame(map(lambda x: x.replace('\n', '').split(','), contents), columns=columns)
    url_list = style_pd.loc[:, ['id', 'url']].values.tolist()
    return style_pd, url_list


def loadProductSearchData(read_path, save_path, p_columns, s_columns):
    '''
    Load the ProductSearch data.
    '''
    p_data = read_path + os.sep + 'P.txt'
    s_data = read_path + os.sep + 'S.txt'
    p_path = save_path + os.sep + 'p_data'
    s_path = save_path + os.sep + 's_data'
    # p train data
    mkdir(save_path)
    mkdir(p_path)
    mkdir(s_path)
    with open(p_data, 'r') as f:
        contents = f.readlines()
    p_pd = pd.DataFrame(map(lambda x: x.replace('\n', '').split(' '), contents), columns=p_columns)
    p_pd['id'] = map(str, range(len(p_pd)))
    cus_url_list = p_pd['cus_url'].values.tolist()
    shop_url_list = p_pd['shop_url'].values.tolist()
    id_list = p_pd['id'].values.tolist()
    all_len = len(id_list)
    for i in xrange(len(id_list)):
        if (divmod(i, 500)[1] == 0) or (i == (-1)):
            print '[{2}] ( {0}/{1} )'.format(i, all_len, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        loadAndSave(cus_url_list[i], id_list[i] + '_cus', p_path)
        loadAndSave(shop_url_list[i], id_list[i] + '_shop', p_path)
    p_pd.to_csv(p_path + os.sep + 'p_pd.csv', index=False)
    # s test data
    with open(s_data, 'r') as f:
        contents = f.readlines()
    s_pd = pd.DataFrame(map(lambda x: x.replace('\n', '').split(' '), contents), columns=s_columns)
    s_pd['id'] = map(str, range(len(s_pd)))
    shop_url_list = s_pd['shop_url'].values.tolist()
    id_list = s_pd['id'].values.tolist()
    all_len = len(id_list)
    for i in xrange(len(id_list)):
        if (divmod(i, 500)[1] == 0) or (i == (-1)):
            print '[{2}] ( {0}/{1} )'.format(i, all_len, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        loadAndSave(shop_url_list[i], id_list[i] + '_shop', s_path)
    s_pd.to_csv(s_path + os.sep + 'p_pd.csv', index=False)


class StyleIdentifySpider(Thread):
    def __init__(self, url, q, save_path, i):
        '''
        Init class StyleIdentifySpider.
        '''
        super(StyleIdentifySpider, self).__init__()
        self.name = url[0]
        self.url = url[1]
        self.q = q
        self.save_path = save_path
        self.contents = None
        self.i = i
    def saveData(self):
        '''
        Save the pictures.
        '''
        print '[{0}] Save the data! {1}'.format(self.i, self.name)
        with open(self.save_path + os.sep + self.name + '.jpg', 'wb') as f:
            f.write(self.contents)
    def getRequest(self):
        '''
        Get the request of url.
        '''
        i = 0
        print '[{0}] getRequest {1}'.format(self.i, self.name)
        time_limit = 1000
        while i <= time_limit:
            try:
                self.contents = requests.get(url=self.url).content
                print '[{0}] getRequest Finish! {1}'.format(self.i, self.name)
                i = time_limit*10
            except:
                time.sleep(0.5)
                i += 1
        if i == (time_limit+1):
            raise Exception(''' url: {0}'''.format(self.url))
    def run(self):
        '''
        Run method.
        '''
        self.getRequest()
        self.saveData()
        self.q.put(self.name)


# 1、风格识别 StyleIdentify
style_identify_read_path = u'D:/SelfLife/Competition/Ai_fashion/时尚风格识别_train/train+val.txt'
style_identify_save_path2 = u'D:/SelfLife/Competition/Ai_fashion/时尚风格识别_train/pictures2'
# 运动,    休闲,  OL/通勤,  日系,    韩版,     欧美,     英伦,  少女,  名媛/淑女,    简约,    自然,  街头/朋克,    民族
# sport,  relax,      ol,  japan,  korean,  america,  england,  girl,       lady,  simple,  nature,       punk,  nation
style_identify_cols = ['id', 'url', 'sport', 'relax', 'ol', 'japan', 'korean', 'america', 'england', 'girl', 'lady', 'simple', 'nature', 'punk', 'nation']

print '[{0}] loadStyleIdentifyData'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
style_pd, url_list = loadStyleIdentifyData(style_identify_read_path, style_identify_cols)

print '[{0}] style_pd.to_csv'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
mkdir(style_identify_save_path2)
style_pd.to_csv(style_identify_save_path2 + os.sep + 'style_pd.csv', index=False)


q = Queue()
Thread_list = []
result_list = []
url_len = len(url_list)
# Thread to get the url data.
print '[{0}] p.start()'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
for i, url in enumerate(url_list):
    p = StyleIdentifySpider(url, q, style_identify_save_path2, i)
    p.start()
    Thread_list.append(p)
    if (divmod(i, 500)[1] == 0) or (i == (url_len-1)):
        time.sleep(10)
# The main thread wait sub thread finish.
for each in Thread_list:
    each.join()
# Get the data in the Queue.
print '[{0}] get.q()'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
while not q.empty():
    result_list.append(q.get())
    tmp_len = len(result_list)
    if (divmod(tmp_len, 500)[1] == 0) or (tmp_len == (url_len-1)):
        print '[{2}] ( {0}/{1} )'.format(tmp_len, url_len, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


# 2、单品搜索 ProductSearch
product_search_read_path = u'D:/SelfLife/Competition/Ai_fashion/时尚单品搜索_train'
product_search_save_path2 = u'D:/SelfLife/Competition/Ai_fashion/时尚单品搜索_train/pictures2'
# columns
product_search_p_cols = ['cus_url', 'cus_left', 'cus_up', 'cus_right', 'cus_down', 'shop_url', 'shop_left', 'shop_up', 'shop_right', 'shop_down', 'type']
product_search_s_cols = ['shop_url', 'shop_left', 'shop_up', 'shop_right', 'shop_down']

# loadProductSearchData(product_search_read_path, product_search_save_path, product_search_p_cols, product_search_s_cols)






