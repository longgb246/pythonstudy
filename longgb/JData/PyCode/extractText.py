#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from collections import Counter
import jieba
import warnings
warnings.filterwarnings('ignore')
import re


def splitPhone(x):
    '''
    种类的提取
    '''
    phone_key = [u'全网通', u'移动', u'联通', u'电信']
    key_str = u''
    for key in phone_key:
        if key in x and key not in key_str:
            key_str = key_str + u' ' + key
    if np.min(map(lambda x: x in key_str, phone_key)) or np.min(map(lambda x: x in key_str, phone_key[1:])) or u'全网通' in key_str:
        key_str = u' 全网通'
    return key_str


def splitColor(x):
    '''
    颜色的提取
    '''
    jieba.add_word(u'X色')
    jieba.add_word(u'卡其色')
    jieba.add_word(u'咖啡色')
    color_list = [u'咖啡色', u'卡其色', u'粉', u'X色', u'金', u'银', u'铜', u'红', u'橙', u'黄', u'绿', u'蓝', u'青', u'紫', u'靛', u'黑', u'褐', u'白', u'灰', u'棕', u'锖']
    x_list = jieba.lcut(x)
    # x_list = x.split(' ')
    drop_list = [u'黄金', u'金立', u'金属', u'金刚', u'金钢', u'金典', u'流金', u'红辣椒', u'红米', u'青玉', u'青春', u'文青', u'青岛', u'青海', u'青漾', u'魅蓝', u'蓝牙', u'蓝魔', u'黑龙江']
    i = 0
    color_str_list = []
    for each in x_list:
        if np.min(map(lambda x: x not in each, drop_list)):
            for color in color_list:
                if color in each and color not in color_str_list:
                    i += 1
                    color_str_list.append(color)
    # 多种颜色处理
    if u'黑' in color_str_list:
        color_str = u'黑'
    elif u'白' in color_str_list:
        color_str = u'白'
    elif len(color_str_list) >= 2:
        color_str = color_str_list[-1]
    elif len(color_str_list) == 1:
        color_str = color_str_list[0]
    else:
        color_str = u''
    # if color_str == u'':
    #     return True
    # else:
    #     return False
    return color_str


def splitGB(x):
    '''
    几G手机、内存、存储
    '''
    # 1、找手机通讯模式
    tmp3 = re.findall(u'{0}手机'.format(r'(\d+G)'), x)
    communication = tmp3[0] if len(tmp3) >= 1 else u''
    memory = u''
    hard = u''
    # 2、标准模式的内存、存储
    tmp = re.findall(r'(\d+G?B?\+\d+G?B?)', x)   # 找到
    # 3、非标准模式的内存、存储
    tmp2 = re.findall(r'(\d+GB?)', x)
    find_3 = 0
    find_2 = 0
    find_4 = 0
    if communication == u'':
        for each in tmp2:
            if each == u'3G':
                find_3 = 1
            if each == u'2G':
                find_2 = 1
            if each == u'4G':
                find_4 = 1
        communication = u'3G' if find_3 == 1 else (u'2G' if find_2 == 1 else (u'4G' if find_4 == 1 else u''))
    if communication not in [u'2G', u'3G', u'4G']:
        communication = u''
    if len(tmp) >= 1:
        num_list = map(lambda x: int(re.findall(r'(\d+)',x)[0]),tmp[0].split(u'+'))
        memory = u'{0}GB'.format(np.min(num_list))
        hard = u'{0}GB'.format(np.max(num_list))
    else:
        try:
            tmp2.remove(communication)
        except:
            pass
        if len(tmp2) == 1:
            tmp2_num = int(re.findall(r'(\d+)', tmp2[0])[0])
            if tmp2_num <= 4:
                memory = u'{0}GB'.format(tmp2_num)
            elif tmp2_num >= 7 and tmp2_num <= 256:
                hard = u'{0}GB'.format(tmp2_num)
        if len(tmp2) >= 2:
            num_list = sorted(map(lambda x: int(re.findall(r'(\d+)',x)[0]), tmp2), reverse=True)
            memory = u'{0}GB'.format(np.min(num_list)) if np.min(num_list) <= 8 else u''
            for each in num_list:
                if each >= 7 and each <= 256:
                    hard = u'{0}GB'.format(np.max(num_list))
                    break
    return [communication, memory, hard]


def splitLength(x):
    '''
    拆分电缆长度
    '''
    units = [u'米', u'm']
    re_com = r'\d+'
    re_com_list = map(lambda x: u'({0}{1})'.format(re_com, x), units)
    re_com_list_str = u'|'.join(re_com_list)
    all_list = map(lambda x: list(x), re.findall(re_com_list_str, x))
    length_list_str = u''
    this_str = u''
    if len(all_list) >= 1:
        all_list = reduce(lambda x,y: x+y, all_list)
        all_list = list(set(map(lambda x:x.replace(u'米', u'm'), all_list)))
        all_list.remove(u'')
        if len(all_list) >= 2:
            length_list_num = map(lambda x: int(re.findall(r'(\d+)m', x)[0]),all_list)
            length_list_str = u'{0}m'.format(np.min(length_list_num))
        elif len(all_list) == 1:
            length_list_str = all_list[0]
    length_list_split = [0, 5, 10, 50, 3600]
    length_list_split_str = ['[{0}m,{1}m)'.format(length_list_split[i], length_list_split[i+1]) for i in range(len(length_list_split)-1)]
    if length_list_str != u'':
        length_list_num = int(re.findall(r'(\d+)m', length_list_str)[0])
        for i, each in enumerate(length_list_split[1:]):
            if length_list_num < each:
                this_str = length_list_split_str[i]
                break
    return this_str


def splitSize(x):
    '''
    电脑尺寸
    '''
    chicun = [u'英寸']
    chicun_str = u'|'.join(map(lambda x: u'({0}){1}'.format(r'\d+\.?\d?', x), chicun))
    big_str = u' '.join(re.findall(chicun_str, x))
    if u'.' not in big_str and len(big_str)==3:
        big_str = big_str[:2]
    if u'.' in big_str:
        big_str = big_str.split(u'.')[0][-2:]
    if big_str != u'':
        big_str += u'英寸'
    return big_str


def splitIDai(x):
    '''
    拆分是I几代的CPU
    '''
    # 规则：剔除 i1、i2、i4。当出现多个的时候，取最大的那个
    iDai = r'(i\d)|(I\d)'
    iDai_find = re.findall(iDai, x)
    iDai_str = u''
    remove_list = [u'', u'i1', u'i2', u'i4']
    if len(iDai_find) >= 1:
        iDai_list = reduce(lambda y,z: list(y)+list(z), iDai_find)
        iDai_list = list(set(map(lambda x: x.replace(u'I', u'i'), iDai_list)))
        for each in remove_list:
            try:
                iDai_list.remove(each)
            except:
                pass
        if len(iDai_list) >= 2:
            iDai_num_list = map(lambda y: int(re.findall(r'i(\d)', y)[0]), iDai_list)
            iDai_list = [u'i{0}'.format(np.max(iDai_num_list))]
        iDai_str = u' '.join(iDai_list)
    return iDai_str


def splitMonitor(x):
    '''
    是否独显
    '''
    monitor = u'独显'
    if monitor in x:
        monitor_str = u'独显'
    else:
        monitor_str = u''
    return monitor_str


def splitGBCom(x):
    '''
    电脑内存的提取
    '''
    # 内存 剔除：5G 7G 9G 3G 10G
    # 硬盘 保留：1000GB 160GB 500GB 256GB 128GB 512GB 120GB 320GB 250GB 64GB 192GB
    #            750GB 32GB 180GB 240GB 60GB 200GB 640GB 80GB
    hard_keep_list = [1000, 160, 500, 256, 128, 512, 120, 320, 250, 64, 192, 750, 32, 180, 240, 60, 200, 640, 80]
    gbCom = r'(\d+GB?)|(\d+TB?)'
    gbCom_list_find = re.findall(gbCom, x)
    memory = u''
    hard = u''
    memory_list = []
    hard_list = []
    has_memory = False
    has_hard = False
    has_TB = False
    if len(gbCom_list_find) >= 1:
        gbCom_list = reduce(lambda x, y: list(x)+list(y), gbCom_list_find)
        gbCom_list = list(set(map(lambda y: y+u'B',map(lambda z: z.replace(u'B', u''), gbCom_list))))
        gbCom_list.remove(u'B')
        if len(gbCom_list)>=1:
            each_num = map(lambda y: int(y[:-2]),gbCom_list)
            for i, each in enumerate(gbCom_list):
                if u'GB' in each and each_num[i]<20 and each_num[i] not in [3, 5, 7, 9, 10]:
                    has_memory = True
                    memory_list.append(each_num[i])
                if u'GB' in each and each_num[i] in hard_keep_list :
                    has_hard = True
                    hard_list.append(each_num[i])
            memory = u'{0}GB'.format(np.max(memory_list)) if has_memory else u''
            hard = u'{0}GB'.format(np.max(hard_list)) if has_hard else u''
            hard_list = []
            for i, each in enumerate(gbCom_list):
                if u'TB' in each and each_num[i] in [1,2]:
                    has_TB = True
                    hard_list.append(each_num[i])
            hard = u'{0}GB'.format(np.max(hard_list)*1024) if has_TB else hard
    range_hard = [32, 128, 256, 500, 640, 2050]
    range_hard_str = ['[ {0}GB, {1}GB )'.format(range_hard[i], range_hard[i+1]) for i in range(len(range_hard)-1)]
    if hard != u'':
        hard_num = int(re.findall(r'(\d+)GB', hard)[0])
        for i, each in enumerate(range_hard[1:]):
            if hard_num < each:
                hard = range_hard_str[i]
                break
    return [memory, hard]


read_path = r'D:\Data\other\JData\2-JData_train'
save_path = r'D:\Data\other\JData\Analysis'
file_name = 'train_sku_basic_info.csv'


if __name__ == '__main__':
    basic = pd.read_table(read_path + os.sep + file_name, encoding='gbk')
    cate3 = list(basic['item_third_cate_cd'].drop_duplicates().values)

    # 【这里按照空格进行分割】
    # =============================================================
    # =                     655 手机
    # =============================================================
    cate = cate3[0]
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.loc[:,['sku_name']]
    text_analysis.index = range(len(text_analysis))

    # 1、双卡双待
    kadai = u'双卡双待'
    count_kadai = map(lambda x: 1 if kadai in x else 0, text_analysis['sku_name'].values)
    text_analysis['kadai'] = count_kadai
    # 2、移动、联通、电信、全网通
    text_analysis['phone_type'] = map(lambda x: splitPhone(x), text_analysis['sku_name'].values)
    # 3、颜色属性
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    # 4、几G、内存、存储
    hard_memory = map(lambda x: splitGB(x), text_analysis['sku_name'].values)
    text_analysis['communication_G'] = map(lambda x: x[0], hard_memory)
    text_analysis['memory_GB'] = map(lambda x: x[1], hard_memory)
    text_analysis['hard_GB'] = map(lambda x: x[2], hard_memory)
    # 5、最终提取结果
    text_analysis.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')

    # # 统计结果的比例
    # mm = Counter(text_analysis['kadai'])
    # mm = Counter(text_analysis['phone_type'])
    # mm = Counter(text_analysis['color'])
    # mm = Counter(text_analysis['communication_G'])
    # mm = Counter(text_analysis['memory_GB'])
    # mm = Counter(text_analysis['hard_GB'])
    # for key, v in sorted(mm.iteritems(), key=lambda x: x[1], reverse=True):
    #     print key, ' : ', v


    # 测试商品：AAAAAA、XXXXX、***********、测试、xxxxx、作废、验证


    # =============================================================
    # =                     1049 电缆
    # =============================================================
    cate = cate3[1]
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.loc[:,['sku_name']]
    text_analysis.index = range(len(text_analysis))

    # 1、颜色提取
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    text_analysis.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')
    # 2、长度提取
    text_analysis['length'] = map(lambda x: splitLength(x), text_analysis['sku_name'].values)

    # # 3、接口【8717个没有，不取】。USB【也是8000多个没有，不取】：结论 - 没有特征了
    # def splitInter(x):
    #     interface = [u'HDMI', u'VGA', u'HUB', u'DVI', u'AUX']
    #     interface_str = u''
    #     for each in interface:
    #         interface_str = interface_str + u' ' + each if each in x else interface_str
    #     return interface_str

    # aa = Counter(map(lambda x: splitInter(x), text_analysis['sku_name'].values))
    # aa_sort = sorted(aa.iteritems(), key=lambda x: x[1], reverse=True)
    # for key, v in aa_sort:
    #     print key, ' : ', v


    # =============================================================
    # =                     672 电脑
    # =============================================================
    cate = cate3[2]
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.loc[:,['sku_name']]
    text_analysis.index = range(len(text_analysis))

    text_analysis.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')

    # 统计词频，找特征。
    # aa = Counter(reduce(lambda x,y: x+y,map(lambda x: jieba.lcut(x), text_analysis['sku_name'].values)))
    # # aa = Counter(reduce(lambda x,y: x+y,map(lambda x: x.split(' '), text_analysis['sku_name'].values)))
    # aa_sort = sorted(aa.iteritems(), key=lambda x: x[1], reverse=True)
    # aa_pd = pd.DataFrame.from_dict(aa_sort)
    # # aa_pd.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')
    # aa_pd.to_csv(save_path + os.sep + 'test_jieba.csv', index=False, encoding='gbk')
    # for key, v in aa_sort:
    #     print key, ' : ', v

    # 统计
    # aa_sorted = sorted(Counter(map(lambda x: splitColor(x), text_analysis['sku_name'].values)).iteritems(), key=lambda x:x[1], reverse=True)
    # aa_sorted = sorted(Counter(map(lambda x: splitSize(x), text_analysis['sku_name'].values)).iteritems(), key=lambda x:x[1], reverse=True)
    # aa_sorted = sorted(Counter(map(lambda x: splitIDai(x), text_analysis['sku_name'].values)).iteritems(), key=lambda x:x[1], reverse=True)
    # for key, v in aa_sorted:
    #     print key, ' : ', v

    # 1、颜色
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    # 2、尺寸
    text_analysis['size'] = map(lambda x: splitSize(x), text_analysis['sku_name'].values)
    # 3、i几
    text_analysis['iDai'] = map(lambda x: splitIDai(x), text_analysis['sku_name'].values)
    # 4、独显
    text_analysis['Monitor'] = map(lambda x: splitMonitor(x), text_analysis['sku_name'].values)
    # 5、内存、存储
    hard_memory = map(lambda x: splitGBCom(x), text_analysis['sku_name'].values)
    text_analysis['memory_GB'] = map(lambda x: x[0], hard_memory)
    text_analysis['hard_GB'] = map(lambda x: x[1], hard_memory)










    # 检验：
    # 1、'item_desc' 是个没什么用的字段
    # basic_used_count = Counter(basic['item_desc'])
    # for key, value in basic_used_count.iteritems():
    #     print key, ' : ', value

    # 2、'data_type' 全部是 1

    # 3、统计品牌的信息
    # brand_count = basic.groupby(['barndname_full'])['item_sku_id_hashed'].count().reset_index().rename(columns={'item_sku_id_hashed':'count'}).sort_values(['count'], ascending=False)
    # brand_count.index = range(len(brand_count))
    # brand_all = pd.DataFrame.from_dict(Counter(basic['barndname_full']), orient='index').reset_index().rename(columns={0:'count', 'index':'brand'}).sort_values(['count'], ascending=False)
    # brand_all.index = range(len(brand_all))

    # 4、对于颜色的统计
    # pd.DataFrame.from_dict(Counter(basic['colour']), orient='index').reset_index().rename(columns={0:'count', 'index':'color'}).sort_values('count', ascending=False)


    # app.fdc_predict_forecast_result
    pass



