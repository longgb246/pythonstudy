#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd


def mapProv(str_code, prov_map):
    '''
    Chinese map to English
    '''
    for each in prov_map:
        if each[0] in str_code:
            return each[1]
    return ''


def readData(read_path, date_limit):
    '''
    read Data
    '''
    date_max = date_limit[-1]
    date_min = date_limit[0]
    sub_paths = os.listdir(read_path)
    all_files = {}
    for sub_path in sub_paths:
        this_path = read_path + os.sep + sub_path
        load_files = os.listdir(this_path)
        for load_file in load_files:
            load_file_split = load_file.split('.')
            if len(load_file_split) == 2:
                this_file = this_path + os.sep + load_file
                this_file_pd = pd.read_csv(this_file, encoding='gbk')
                this_date_min = min(map(lambda x:str(x), this_file_pd.columns[1:]))
                this_date_max = max(map(lambda x:str(x), this_file_pd.columns[1:]))
                if this_date_max >= date_max and this_date_min <= date_min:
                    pass
                else:
                    continue
                try:
                    this_file_pd['province'] = map(lambda x: mapProv(x, prov_map), this_file_pd['province'].values)
                    this_file_pd = order_pd.merge(this_file_pd, on=['province'], how='left').sort_values(['order'])
                    this_file_pd = this_file_pd.fillna(0)
                except:
                    print load_file_split[0]
                    print this_file_pd
                all_files[load_file_split[0]] = this_file_pd
    return all_files


def transData(all_data):
    '''
    trans Data format
    '''
    all_data_trans = []
    for name, this_data in all_data.iteritems():
        data_columns = list(this_data.columns)
        try:
            data_columns.remove('province')
            data_columns.remove('order')
        except:
            raise Exception
        tran_data = []
        for tran_col in data_columns:
            tmp_data = this_data.loc[:, ['province', 'order', tran_col]]
            tmp_data = tmp_data.rename(columns={tran_col : name})
            tmp_data['year'] = str(tran_col)
            tran_data.append(tmp_data)
        this_trans_data = pd.concat(tran_data, axis=0).sort_values(['year', 'order'])
        this_trans_data.index = range(len(this_trans_data))
        all_data_trans.append(this_trans_data)
    return all_data_trans


def getVarName(keep_data):
    dependent_var = ['FiscalTransparency']
    independent_var = ['MarketizationIndex', 'ProvincialFinancialStatisticsExpenditure']
    control_var_drop = ['province', 'order', 'year'] + dependent_var + independent_var
    control_var = list(keep_data.columns)
    for drop_var in control_var_drop:
        try:
            control_var.remove(drop_var)
        except:
            pass
    return dependent_var, independent_var, control_var


def standardData(data, cols):
    '''
    standardization
    '''
    for col in cols:
        avg_v = np.mean(data[col])
        std_v = np.std(data[col])
        data[col] = (data[col] - avg_v)/std_v
    return data


read_path = r'C:\Users\longguangbin\Desktop\Data_Code\origin_data'
save_path = r'C:\Users\longguangbin\Desktop\Data_Code\arange_data'

prov_map = [[u'安徽',	'Anhui'],
            [u'北京',	'Beijing'],
            [u'重庆',	'Chongqing'],
            [u'福建',	'Fujian'],
            [u'甘肃',	'Gansu'],
            [u'广东',	'Guangdong'],
            [u'广西',	'Guangxi'],
            [u'贵州',	'Guizhou'],
            [u'海南',	'Hainan'],
            [u'河北',	'Hebei'],
            [u'黑龙江',	'Heilongjiang'],
            [u'河南',	'Henan'],
            [u'湖北',	'Hubei'],
            [u'湖南',	'Hunan'],
            [u'江苏',	'Jiangsu'],
            [u'江西',	'Jiangxi'],
            [u'吉林',	'Jilin'],
            [u'辽宁',	'Liaoning'],
            [u'内蒙古',	'NeiMongol'],
            [u'宁夏',	'NingxiaHui'],
            [u'青海',	'Qinghai'],
            [u'陕西',	'Shaanxi'],
            [u'山东',	'Shandong'],
            [u'上海',	'Shanghai'],
            [u'山西',	'Shanxi'],
            [u'四川',	'Sichuan'],
            [u'天津',	'Tianjin'],
            [u'新疆',	'XinjiangUygur'],
            [u'西藏',	'Xizang'],
            [u'云南',	'Yunnan'],
            [u'浙江',	'Zhejiang']]
order_pd = pd.DataFrame(zip(map(lambda x: x[1],prov_map), range(len(prov_map))), columns=['province', 'order'])
date_limit = map(lambda x: str(x), range(2008, 2015))

all_data = readData(read_path, date_limit)
all_data_trans = transData(all_data)

keep_data = reduce(lambda x, y: x.merge(y, on=['province', 'order', 'year']),all_data_trans)
dependent_var, independent_var, control_var = getVarName(keep_data)

keep_col = ['year', 'province'] + dependent_var + independent_var + control_var
keep_data.loc[:, keep_col].to_csv(save_path + os.sep + 'arange_data.csv', index=False)

keep_data_center = keep_data.copy()
keep_data_center = standardData(keep_data_center, dependent_var + independent_var + control_var)
keep_data_center.loc[:, keep_col].to_csv(save_path + os.sep + 'arange_data_standard.csv', index=False)

# For Corr
keep_data.loc[:, dependent_var + independent_var + control_var].to_csv(save_path + os.sep + 'arange_data_corr.csv', index=False)

# For MD
keep_data_md = keep_data.groupby(['province']).mean().reset_index()
keep_data_md = pd.concat([keep_data_md.iloc[:20,:], pd.DataFrame(np.zeros([1,len(keep_data_md.columns)]), columns=keep_data_md.columns), keep_data_md.iloc[20:,:]])
keep_data_md.to_csv(save_path + os.sep + 'arange_data_md.csv', index=False)

# FiscalTransparency
# MarketizationIndex
# ProvincialFinancialStatisticsExpenditure
# + LocalFiscalTaxRevenue
# + UrbanPopulationDensity
# + ManyPerCapitaUrbanRoadArea
# + TotalInvestmentOfForeignInvestedEnterprises
# + ManyBasicOilReserves
# + ManyPermanentPopulation
# + AverageWageOfStateOwnedUnit
# + ProvincialFinancialStatisticsIncome
# + GovernmentScaleRegionalGrossDomesticProduct
# + ProvincialFinancialStatisticsIncomePre
# + ManyDeathRate
# + ManyBirthRate
# + ManyCountyDivisionNumber
# + LocalFiscalRevenue
# + ManyBasicCoalReserves
# + ManyPrefectureLevelDivisionNumber
# + EducationLevelOfResidents
# + ManyPrefectureLevelCity
# + GovernmentScaleExpenditure
# + ManyBasicReservesOfNaturalGas

