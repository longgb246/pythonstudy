#-*- coding:utf-8 -*-
import os
import time
import requests
import pandas as pd


def mkdir(mkdir_path):
    '''
    Make dir.
    '''
    if not os.path.exists(mkdir_path):
        os.mkdir(mkdir_path)


def loadAndSave(url, name, save_path):
    '''
    Load the picture and save it.
    '''
    url_contents = requests.get(url)
    with open(save_path + os.sep + name + '.jpg', 'wb') as f:
        f.write(url_contents.content)


def loadStyleIdentifyData(read_path, save_path, columns):
    '''
    Load the StyleIdentify data.
    '''
    print 'loadStyleIdentifyData Start!'
    with open(read_path, 'r') as f:
        contents = f.readlines()
    style_pd = pd.DataFrame(map(lambda x: x.replace('\n', '').split(','), contents), columns=columns)
    mkdir(save_path)
    url_list = style_pd['url'].values.tolist()
    name_list = style_pd['id'].values.tolist()
    all_len = len(url_list)
    for i, each_url in enumerate(url_list):
        if (divmod(i, 500)[1] == 0) or (i == (-1)):
            print '[{2}] ( {0}/{1} )'.format(i, all_len, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        loadAndSave(each_url, name_list[i], save_path)
    style_pd.to_csv(save_path + os.sep + 'style_pd.csv', index=False)
    print 'loadStyleIdentifyData Finish!'


def loadProductSearchData(read_path, save_path, p_columns, s_columns):
    '''
    Load the ProductSearch data.
    '''
    print 'loadProductSearchData Start!'
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
    print 'loadProductSearchData Finish!'


# 1、风格识别 StyleIdentify
style_identify_read_path = u'D:/SelfLife/Competition/Ai_fashion/时尚风格识别_train/train+val.txt'
style_identify_save_path = u'D:/SelfLife/Competition/Ai_fashion/时尚风格识别_train/pictures'
# 运动,    休闲,  OL/通勤,  日系,    韩版,     欧美,     英伦,  少女,  名媛/淑女,    简约,    自然,  街头/朋克,    民族
# sport,  relax,      ol,  japan,  korean,  america,  england,  girl,       lady,  simple,  nature,       punk,  nation
style_identify_cols = ['id', 'url', 'sport', 'relax', 'ol', 'japan', 'korean', 'america', 'england', 'girl', 'lady', 'simple', 'nature', 'punk', 'nation']

loadStyleIdentifyData(style_identify_read_path, style_identify_save_path, style_identify_cols)

# 2、单品搜索 ProductSearch
product_search_read_path = u'D:/SelfLife/Competition/Ai_fashion/时尚单品搜索_train'
product_search_save_path = u'D:/SelfLife/Competition/Ai_fashion/时尚单品搜索_train/pictures'
# columns
product_search_p_cols = ['cus_url', 'cus_left', 'cus_up', 'cus_right', 'cus_down', 'shop_url', 'shop_left', 'shop_up', 'shop_right', 'shop_down', 'type']
product_search_s_cols = ['shop_url', 'shop_left', 'shop_up', 'shop_right', 'shop_down']

loadProductSearchData(product_search_read_path, product_search_save_path, product_search_p_cols, product_search_s_cols)

