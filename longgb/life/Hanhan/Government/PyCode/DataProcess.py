#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import statsmodels.api as sm
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def mapProv(str_code, prov_map):
    '''
    Chinese map to English
    '''
    for each in prov_map:
        if each[0] in str_code:
            return each[1]
    return ''


def readData(read_path, date_limit, prov_map, order_pd):
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
    '''
    Get the Variable Char Name
    '''
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


def getStatIndex(data, func, func_name, cols):
    '''
    cal the index of stat
    '''
    tmp_list = []
    for col in cols:
        this_data = data[col]
        tmp_list.append([col, func(this_data)])
    tmp_data = pd.DataFrame(tmp_list, columns = ['Variable', func_name])
    return tmp_data


def getDescripStat(keep_data, stat_cols):
    '''
    get the Descriptive statistic
    '''
    get_stat = [[np.mean, 'mean'], [np.median, 'median'], [np.std, 'std'], [np.min, 'min'], [np.max, 'max'],
                [lambda x: np.mean(x)/np.std(x), 'cv'], [len,'Obs']]
    stat_list = []
    for stat_index in get_stat:
        stat_list.append(getStatIndex(keep_data, stat_index[0], stat_index[1], stat_cols))
    descrip_stat = reduce(lambda x, y: x.merge(y, on=['Variable']), stat_list)
    return descrip_stat


def plotPairPlot(save_path, keep_data, cols):
    '''
    Plot the Scatter Plot
    '''
    a = sns.pairplot(keep_data.loc[:, cols])
    a.savefig(save_path + os.sep + 'corr_scatter_var.pdf', format='pdf')


def logData(data, cols):
    '''
    Log the Data
    '''
    for col in cols:
        data[col] = np.log(data[col])
    return data


def rmDataName(control_var, cols):
    '''
    remove the Data Name
    '''
    for col in cols:
        try:
            control_var.remove(col)
        except:
            pass
    return control_var


def main():
    # Set the path
    read_path = r'D:\SelfLife\HanHan\Data_Code\data_origin'
    save_path = r'D:\SelfLife\HanHan\Data_Code\data_arange'
    save_stat_path = r'D:\SelfLife\HanHan\Data_Code\results'

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

    all_data = readData(read_path, date_limit, prov_map, order_pd)
    all_data_trans = transData(all_data)

    keep_data = reduce(lambda x, y: x.merge(y, on=['province', 'order', 'year']),all_data_trans)
    dependent_var, independent_var, control_var = getVarName(keep_data)

    keep_data['GovernmentScaleExpenditurePre'] = keep_data['GovernmentScaleExpenditure'] / keep_data['GovernmentScaleRegionalGrossDomesticProduct']
    control_var = control_var + ['GovernmentScaleExpenditurePre']
    need_log = [ 'LocalFiscalTaxRevenue',
                 'UrbanPopulationDensity',
                 'TotalInvestmentOfForeignInvestedEnterprises',
                 'ManyPermanentPopulation',
                 'AverageWageOfStateOwnedUnit',
                 'ProvincialFinancialStatisticsIncome',
                 'GovernmentScaleRegionalGrossDomesticProduct',
                 'ProvincialFinancialStatisticsIncomePre',
                 'LocalFiscalRevenue',
                 'EducationLevelOfResidents',
                 'GovernmentScaleExpenditure']
    need_rm = ['GovernmentScaleRegionalGrossDomesticProduct', 'GovernmentScaleExpenditure',
               'ManyBasicOilReserves', 'ManyBasicReservesOfNaturalGas', 'ManyBasicCoalReserves',
               'ManyPrefectureLevelDivisionNumber','ManyCountyDivisionNumber']

    # Log the Data
    keep_data = logData(keep_data, need_log)
    control_var = rmDataName(control_var, need_rm)

    patch_map = {
        'FiscalTransparency' : 'FT',
        'MarketizationIndex' : 'Institution',
        'ProvincialFinancialStatisticsExpenditure' : 'GovComp',
        'ProvincialFinancialStatisticsIncome' : 'FinancialIncome',
        'ManyBirthRate' : 'BirthRate',
        'ProvincialFinancialStatisticsIncomePre' : 'FinancialIncomePer',
        'ManyPrefectureLevelCity' : 'PrefectureLevelCity',
        'GovernmentScaleExpenditurePre' : 'Governmentsize',
        'TotalInvestmentOfForeignInvestedEnterprises' : 'TotInvestOfForeign',
        'AverageWageOfStateOwnedUnit' : 'AvgWageSO',
        'EducationLevelOfResidents' : 'EduLevelOfResidents'
    }

    for key, value in patch_map.iteritems():
        keep_data[value] = keep_data[key]

    # Step Regression in R : Results
    dependent_var = ['FT']
    independent_var = ['Institution', 'GovComp']
    control_var = ['FinancialIncome',
                   'BirthRate',
                   'FinancialIncomePer',
                   'PrefectureLevelCity',
                   'Governmentsize',
                   'TotInvestOfForeign',
                   'AvgWageSO',
                   'EduLevelOfResidents']

    # LocalFiscalTaxRevenue   log
    # UrbanPopulationDensity    log
    # TotalInvestmentOfForeignInvestedEnterprises   log
    # ManyPermanentPopulation       log
    # AverageWageOfStateOwnedUnit   log
    # ProvincialFinancialStatisticsIncome       log
    # ProvincialFinancialStatisticsIncomePre    log
    # LocalFiscalRevenue            log
    # EducationLevelOfResidents     log
    # ManyBasicOilReserves      [ rm ]
    # GovernmentScaleRegionalGrossDomesticProduct        [ rm ]
    # GovernmentScaleExpenditure    [ rm ]
    # ManyBasicReservesOfNaturalGas     [ rm ]


    # Descriptive statistical analysis
    descrip_stat = getDescripStat(keep_data, dependent_var + independent_var + control_var)
    descrip_stat.to_csv(save_stat_path + os.sep + 'describe_stat.csv', index=False)

    # Plot the Scatter Plots
    plotPairPlot(save_stat_path, keep_data, dependent_var + independent_var + control_var)

    # Save Data
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

    # Print Regression
    mod = sm.OLS(keep_data[dependent_var], keep_data.loc[:, independent_var + control_var])
    res = mod.fit()
    print control_var
    print res.summary()

    # Robustness test Regression
    print '-----------------------------------------------------------'
    print '                 Robustness test Regression'
    print '-----------------------------------------------------------'
    independent_var = ['Institution', 'TotInvestOfForeign']
    control_var = ['FinancialIncome',
                   'BirthRate',
                   'FinancialIncomePer',
                   'PrefectureLevelCity',
                   'Governmentsize',
                   'AvgWageSO',
                   'EduLevelOfResidents']
    mod = sm.OLS(keep_data[dependent_var], keep_data.loc[:, independent_var + control_var])
    res = mod.fit()
    print res.summary()

    # res.rsquared        # 调整后的 R 方
    # res.pvalues         # P 值
    # res.params          # 回归结果


    # + FT
    # + Institution
    # + GovComp
    # + FinancialIncome                                 # 地方财政统计总收入
    # + BirthRate                                       # 出生率
    # + FinancialIncomePer                              # 地方财政统计人均收入
    # + PrefectureLevelCity                             # 地级区划数
    # + Governmentsize                                  # 政府规模财政支出
    # + TotInvestOfForeign                              # 外商投资企业投资总额
    # + AvgWageSO                                       # 国有单位平均工资
    # + EduLevelOfResidents                             # 居民受教育程度


if __name__ == '__main__':
    main()


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
# + GovernmentScaleExpenditurePre


# + LocalFiscalTaxRevenue
# + UrbanPopulationDensity
# + ManyPerCapitaUrbanRoadArea
# + TotalInvestmentOfForeignInvestedEnterprises
# + ManyPermanentPopulation
# + AverageWageOfStateOwnedUnit
# + ProvincialFinancialStatisticsIncome
# + ProvincialFinancialStatisticsIncomePre
# + ManyDeathRate
# + ManyBirthRate
# + LocalFiscalRevenue
# + EducationLevelOfResidents
# + ManyPrefectureLevelCity
# + GovernmentScaleExpenditurePre


