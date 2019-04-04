#-*- coding:utf-8 -*-
#===============================================================================
#
#         FILE: extractText.py
#
#  DESCRIPTION: 用于提取文本信息
#      OPTIONS: ---
# REQUIREMENTS: ---
#        NOTES: ---
#       AUTHOR: lgb453476610@163.com
#      COMPANY: self
#      VERSION: 1.0
#      CREATED: 2017-09-04
#===============================================================================
import os
import numpy as np
import pandas as pd
from collections import Counter
import jieba
import time
import warnings
warnings.filterwarnings('ignore')
import re


# ==========================================================
# =                     功能函数
# ==========================================================
def printRunTime(t1, name=""):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if name != "":
        name = " ( " + name + " )"
    if hor_d >0:
        print '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


# ==========================================================
# =                     应用函数
# ==========================================================
def splitTestItems(x):
    '''
    剔除测试数据
    '''
    test_list = [u'AAAA', u'XXXX', u'****', u'测试专用', u'测试商品', u'xxxx', u'作废', u'验证']
    for each in test_list:
        if each in x:
            return False
    return True


def splitPhone(x):
    '''
    手机：种类的提取
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
    手机、电脑、电缆：颜色的提取
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
    if color_str in [u'咖啡色', u'卡其色']:
        color_str = u'棕'
    if color_str == u'锖':
        color_str = u'青'
    return color_str


def splitGB(x):
    '''
    手机：几G手机、内存、存储
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
    电缆：拆分电缆长度
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
    电脑：电脑尺寸
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
    电脑：拆分是I几代的CPU
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
    电脑：是否独显
    '''
    monitor = u'独显'
    if monitor in x:
        monitor_str = u'独显'
    else:
        monitor_str = u''
    return monitor_str


def splitGBCom(x):
    '''
    电脑：电脑内存的提取
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


def splitBrand(x_list):
    '''
    手机壳：品牌提取
    { 比例 } 手机壳:
        苹果  :  4315
        华为  :  1282
          :  921
        小米  :  747
        三星  :  648
        魅族  :  475
        OPPO  :  286
        vivo  :  196
        多品牌  :  84
        锤子  :  28
    '''
    x = x_list[0]
    y = x_list[1]
    brand_list = [u'iPhone', u'苹果', u'OPPO', u'华为', u'vivo', u'小米', u'三星', u'魅族', u'锤子']
    str_list = []
    for each in brand_list:
        if (each in x) or (each in y) and (each not in str_list):
            if each == u'iPhone':
                str_list.append(u'苹果')
            else:
                str_list.append(each)
    str_list = list(set(str_list))
    if len(str_list) >= 2:
        brand_str = u'多品牌'
    elif len(str_list) == 1:
        brand_str = str_list[0]
    else:
        brand_str = u''
    return brand_str


def splitShell(x):
    '''
    手机壳：是硬壳-软壳
    { 比例 }
              :  4597
        软壳  :  3012
        硬壳  :  1373
    '''
    shell_list = [u'硬壳', u'软壳']
    str_list = []
    for each in shell_list:
        if each in x and each not in str_list:
            str_list.append(each)
    shell_str = u' '.join(str_list)
    return shell_str


def splitProThrow(x):
    '''
    手机壳：是否防摔
    { 比例 }
          :  4787
        防摔  :  4195
    '''
    throw = u'防摔'
    str_list = []
    throw_str = u'防摔' if throw in x else u''
    return throw_str


def splitSingle(x, arr_str):
    '''
    手机膜：属性单一的变量提取
    '''
    arr_str = arr_str if arr_str in x else u''
    return arr_str


# read_path = r'D:\Data\other\xxxata\2-xxxata_train'
# save_path = r'D:\Data\other\xxxata\Analysis'
# file_name = 'train_sku_basic_info.csv'
read_path = r'D:\Data\other\xxxata\Run'
save_path = r'D:\Data\other\xxxata\Run\Results'
file_name = 'cleaned_first_sales_df.csv'


# item_sku_id_hashed,main_sku_id_hashed,sku_name,item_first_cate_cd,item_second_cate_cd,item_third_cate_cd,brand_code,barndname_cn,shelves_dt,colour,first_sale_date,first_field_keshou_date,first_real_keshou_date,total_7days_sales,sales_beyond0_days,field_keshou_days,real_keshou_days,avg_keshou_sales


if __name__ == '__main__':
    basic = pd.read_table(read_path + os.sep + file_name, encoding='gbk', sep=',')
    # basic = pd.read_table(read_path + os.sep + file_name, encoding='gbk')
    cate3 = list(basic['item_third_cate_cd'].drop_duplicates().values)

    # 剔除 测试商品
    # 字段：AAAAAA、XXXXX、***********、测试专用、测试商品、xxxxx、作废、验证
    # sku：一共 23004 个，剔除 5 个
    test = basic.loc[map(lambda x: splitTestItems(x), basic['sku_name'].values), :]

    # =============================================================
    # =                     655 手机
    # =============================================================
    # 2713 个 sku
    t1 = time.time()
    cate = 655
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.copy()
    text_analysis.index = range(len(text_analysis))
    # 1、双卡双待
    kadai = u'双卡双待'
    text_analysis['kadai'] = map(lambda x: 1 if kadai in x else 0, text_analysis['sku_name'].values)
    # 2、移动、联通、电信、全网通
    text_analysis['phone_type'] = map(lambda x: splitPhone(x), text_analysis['sku_name'].values)
    # 3、颜色属性
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    # 4、几G、内存、存储
    hard_memory = map(lambda x: splitGB(x), text_analysis['sku_name'].values)
    text_analysis['communication_G'] = map(lambda x: x[0], hard_memory)
    text_analysis['memory_GB'] = map(lambda x: x[1], hard_memory)
    text_analysis['hard_GB'] = map(lambda x: x[2], hard_memory)

    # 提取结果
    text_analysis.to_csv(save_path + os.sep + 'sku_basic_info_655.csv', index=False, encoding='gbk')
    printRunTime(t1, ' 655 手机 ')


    # =============================================================
    # =                     1049 电缆
    # =============================================================
    # 5698 个 sku
    t1 = time.time()
    cate = 1049
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.copy()
    text_analysis.index = range(len(text_analysis))
    # 1、颜色提取
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    text_analysis.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')
    # 2、长度提取
    text_analysis['length'] = map(lambda x: splitLength(x), text_analysis['sku_name'].values)

    # 提取结果
    text_analysis.to_csv(save_path + os.sep + 'sku_basic_info_1049.csv', index=False, encoding='gbk')
    printRunTime(t1, ' 1049 电缆 ')


    # =============================================================
    # =                     672 电脑
    # =============================================================
    # 2068 个 sku
    t1 = time.time()
    cate = 672
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.copy()
    text_analysis.index = range(len(text_analysis))
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

    # 提取结果
    text_analysis.to_csv(save_path + os.sep + 'sku_basic_info_672.csv', index=False, encoding='gbk')
    printRunTime(t1, ' 672 电脑 ')

    # =============================================================
    # =                     866 手机壳
    # =============================================================
    # 8982 个 sku
    t1 = time.time()
    cate = 866
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.copy()
    text_analysis.index = range(len(text_analysis))
    # 1、牌子  iPhone  苹果  OPPO  华为  vivo  小米  三星  魅族
    # 【逻辑】：先是sku_name、或者brand里面包含上面品牌
    text_analysis['brand'] = map(lambda x: splitBrand(x), text_analysis.loc[:,['sku_name', 'barndname_cn']].values)
    # 2、颜色
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    # 3、硬壳 - 软壳
    text_analysis['shell'] = map(lambda x: splitShell(x), text_analysis['sku_name'].values)
    # 4、防摔
    text_analysis['proThrow'] = map(lambda x: splitProThrow(x), text_analysis['sku_name'].values)

    # 提取结果
    text_analysis.to_csv(save_path + os.sep + 'sku_basic_info_866.csv', index=False, encoding='gbk')
    printRunTime(t1, ' 866 手机壳 ')


    # =============================================================
    # =                     867 手机膜
    # =============================================================
    # 3543 个 sku
    t1 = time.time()
    cate = 867
    basic_tmp = basic[basic['item_third_cate_cd']==cate]
    text_analysis = basic_tmp.copy()
    text_analysis.index = range(len(text_analysis))
    # 1、钢化
    arr_str = u'钢化'
    text_analysis['steel'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 2、玻璃
    arr_str = u'玻璃'
    text_analysis['glass'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 3、防爆
    arr_str = u'防爆'
    text_analysis['crack'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 4、高清
    arr_str = u'高清'
    text_analysis['highDefine'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 5、蓝光
    arr_str = u'蓝光'
    text_analysis['blue'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 6、3D
    arr_str = u'3D'
    text_analysis['3D'] = map(lambda x: splitSingle(x, arr_str), text_analysis['sku_name'].values)
    # 7、颜色
    text_analysis['color'] = map(lambda x: splitColor(x), text_analysis['sku_name'].values)
    # 8、品牌
    text_analysis['brand'] = map(lambda x: splitBrand(x), text_analysis.loc[:,['sku_name', 'barndname_cn']].values)

    # 提取结果
    text_analysis.to_csv(save_path + os.sep + 'sku_basic_info_867.csv', index=False, encoding='gbk')
    printRunTime(t1, ' 867 手机膜 ')


    print 'Task have been Finished !! '


    # # 1、输出 分词统计 观测
    # aa = Counter(reduce(lambda x,y: x+y,map(lambda x: jieba.lcut(x), text_analysis['sku_name'].values)))
    # aa = Counter(reduce(lambda x,y: x+y,map(lambda x: x.split(' '), text_analysis['sku_name'].values)))
    # aa_sort = sorted(aa.iteritems(), key=lambda x: x[1], reverse=True)
    # aa_pd = pd.DataFrame.from_dict(aa_sort)
    # aa_pd.to_csv(save_path + os.sep + 'test.csv', index=False, encoding='gbk')
    # aa_pd.to_csv(save_path + os.sep + 'test_jieba.csv', index=False, encoding='gbk')

    # # 2、统计
    # aa_sorted = sorted(Counter(map(lambda x: splitColor(x), text_analysis['sku_name'].values)).iteritems(), key=lambda x:x[1], reverse=True)
    # for key, v in aa_sorted:
    #     print key, ' : ', v

    # 金 白 黑 银 灰 蓝 粉 红 绿 紫 橙 黄 青(锖) 棕(卡其色 咖啡色)


    # # 3、剔除异常
    # yichang = text_analysis.loc[map(lambda x: splitShell(x), text_analysis['sku_name'].values),:]


    # ==================================================================
    # =                         其他结论
    # ==================================================================
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
    def transforVar(df_list, column_name):
        for df in df_list:
            values_name = df.loc[:, [column_name]].drop_duplicates()
            int_str = map(lambda x: str(x),range(20))
            categories = {}
            for i, each in enumerate(values_name):
                categories[each] = int_str[i]
            df[column_name] = df[column_name].applymap(categories.get)
        return df_list


