# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def uppath(n=1):
    # __file__ = r'D:\Lgb\pythonstudy\longgb\analysis\analysis_run.py'
    if n == 0:
        return os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), (os.pardir + os.sep) * (n - 1) + os.pardir))

sys.path.append(uppath(4))  # 当前文件所在目录的上4级目录。 [analysis]-pbs-jd-com-src，定位到src目录。
# 标准包导入
# from com.jd.pbs.analysis import calKpi
# from com.jd.pbs.analysis import plotData
# from com.jd.pbs.analysis import report as rep

# test包路径导入
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysis import calKpi
from analysis import plotData
from analysis import report as rep

"""
使用说明：
修改file_path为数据文件所在路径，以及研究时间段sim_date_range，然后run。

结果说明：
建立新的RealityAnalysis文件夹，包含
1）kpi文件和z_value_frame文件，前者包含每个sku的现货率、周转等信息，后者包含每个采购单-SKU维度的补货信息，包括z,bp,vlt。
2）report文件夹，包含各种kpi、补货相关的图，以及4种现货-周转情况的sample，位于各自文件夹。还有一个简单的html报告文件
3）其中上柜日期目前还使用的是otc_tm字段，后面数据源修改后用新的字段
"""
# ===============================================================================
# =                                 0、参数设置                                  =
# ===============================================================================
this_path = uppath(0)
# file_path = r'D:\Lgb\ReadData\01_analysis_report'
file_path = uppath(1) + os.sep + 'wenjian' + os.sep + 'Readdata'
data_name = 'data11977.csv'
sim_date_range = ['2016-07-01', '2016-10-24']
pinlei = [u'厨具',u'厨房配件',u'储物/置物架']
pinleiid = 11977
# 仿真
# sim_path = r'D:\tmp\simulatePrograme\SkuSimulationPbs\2016-11-23_17-45-52\simulation_results'
sim_path = file_path

# 读入数据
data_origin = pd.read_csv(file_path + '\\' + data_name, sep='\t')
name = data_origin.item_third_cate_name.iloc[0]  # item_third_cate_name：商品三级分类名称
sort_id = data_origin.item_third_cate_cd.iloc[0]  # item_third_cate_cd：商品三级分类代码

# 统计一些基本数量
data_origin.day_string = pd.to_datetime(data_origin.day_string)  # day_string：日期
total_sku = data_origin.item_sku_id.unique().size  # item_sku_id：SKU的ID

# shelves_dt：上架时间
# 挑选上架时间在7.1日前
data = data_origin[pd.to_datetime(data_origin.shelves_dt) <= pd.to_datetime(sim_date_range[0])]  # 去掉上架时间在7月1日之后的，先假定这个字段是正确的
screen_sku = data.item_sku_id.unique().size

stats1 = {"total": total_sku, "screened": screen_sku}

# 按照时间再筛一次，data作为后面KPI分析等的数据源
# 挑选时间在7.1-10.24日之间
data = data.drop(data[(data.day_string < sim_date_range[0]) | (data.day_string > sim_date_range[1])].index)

first_day_ofdsales = {}
first_day_inv = {}
total_sales = {}

# ofdsales：预测重写日销量（后28天的预测量）。variance：预测日标准差。
# stock_qtty：库存数量。
# total_sales：总销量。
data_group = data.groupby('item_sku_id')
for item_sku_id, group in data_group:
    first_day_ofdsales[item_sku_id] = (not isinstance(group.ofdsales.iloc[0], str)) or (not isinstance(group.variance.iloc[0], str))
    first_day_inv[item_sku_id] = np.isnan(group.stock_qtty.iloc[0])
    total_sales[item_sku_id] = (abs(np.sum(group.total_sales) - 0) <= 1e-3)
sim_screen = pd.DataFrame.from_dict([first_day_ofdsales, first_day_inv, total_sales]).T
sim_screen.columns = ['first_day_ofdsales', 'first_day_inv', 'total_sales']
# 三个条件，首日预测日销量、标准差为str；库存不为空；总销量为0。False为都不满足，True为一个满足就可以。
sim_screen['three_conditions'] = sim_screen.apply(lambda row: (row[0] or row[1] or row[2]), axis=1)

#####  现状分析
# 建立输出路径
analysis_path = file_path + '\\RealityAnalysis'
if not os.path.exists(analysis_path):
    os.mkdir(analysis_path)

# 供应商
# gongyingshangid = [2940635]
# weneed = data[map(lambda x:x in gongyingshangid, data["item_sku_id"].values)]
# weneed2 = weneed.loc[:,["item_sku_id","supp_name"]]
# weneed3 = weneed2.drop_duplicates()
# weneed3.to_csv(file_path + '\\RealityAnalysis'+ os.sep + "gongying3.csv")
tempnum = data.item_sku_id.unique().size - sim_screen.three_conditions.sum()
print "SKU总数为{0}个，在分析首日之前上柜的SKU总数为{1}个：".format(total_sku, screen_sku)
print "1）首日销量预测为空的sku数量为：%d" % sim_screen.first_day_ofdsales.sum()
print "2）首日库存数据为0的sku数量为：%d" % sim_screen.first_day_inv.sum()
print "3）期间总销量为0的sku数量为：%d" % sim_screen.total_sales.sum()
print "以上情况至少出现一次的sku数量为{0}个，满足筛选条件sku个数为{1}个（占总数{2:.2f}%）用于仿真。".format(
    sim_screen.three_conditions.sum(), tempnum, tempnum / total_sku * 100)


# ===============================================================================
# =                                 1、描述性统计分析                            =
# ===============================================================================
count_supp_num = data[data.supp_brevity_cd.isnull() == False]["supp_brevity_cd"].drop_duplicates().count()
count_caigou = data[data.pur_bill_id.isnull() == False]["pur_bill_id"].count()
print "1.sku总数为：{0}个".format(screen_sku)
print "2.分析期间总采购次数为：{0}次".format(count_caigou)
print "3.分析期间供应商总数为：{0}个".format(count_supp_num)


# ===============================================================================
# =                                 2、kpi分析                                  =
# ===============================================================================
print "计算 KPI..."
kpi_frame = calKpi.calcKpi(data, analysis_path)
# 2.1、现货率
mean_cr = np.nanmean(kpi_frame.CR)
median_cr = np.nanmedian(kpi_frame.CR)
count_0 = Counter(kpi_frame.CR == 0)[True]
count_09 = Counter(kpi_frame.CR > 0.9)[True]
temp_str = u"现货率均值为{0:.2f}%，中位数{1:.2f}%。其中现货率大于90%的sku个数为{2}，占总数（{3}个）的{4:.2f}%，现货率为0的sku有{5}个，即分析数据时间区间内库存一直为0。"
kpi_cr_str = temp_str.format(mean_cr * 100, median_cr * 100, count_09, screen_sku, count_09/screen_sku*100, count_0)
print kpi_cr_str
# 2.2、周转分析
mean_cr = np.nanmean(kpi_frame.TD)
median_cr = np.nanmedian(kpi_frame.TD)
count_60 = Counter(kpi_frame.TD < 60)[True]
temp_str = u"周转的均值为{0:.0f}天，中位数为{1:.0f}天。周转小于60天（{2:.0f}个sku，{3:.2f}%），周转大于60天（{4:.0f}个sku，{5:.2f}%）。"
kpi_ito_str = temp_str.format(mean_cr, median_cr, count_60, count_60/screen_sku*100, screen_sku - count_60, (1-(count_60/screen_sku))*100)
print kpi_ito_str


# ===============================================================================
# =                                 3、计算Z值                                  =
# ===============================================================================
print "计算 Z..."
# 补货点分析，z_value_frame不止z_value，有每一条采购单的信息
z_value_frame = calKpi.calcZ(data, analysis_path)


# ===============================================================================
# =                                 4、供应商分析                                =
# ===============================================================================
print "供应商分析..."
z_value_frame.supp_brevity_cd.unique()
supp_value_frame = calKpi.calcsupp(data, analysis_path)


# ===============================================================================
# =                                 5、统一画图                                  =
# ===============================================================================
# 采购
print "plot 采购图..."
plotData.plotcaigou(data, analysis_path)
# KPI
# reload(plotData)
print "plot KPI..."
plotData.plotkpi(kpi_frame, analysis_path)
# Z值
print "plot Z值..."
z_output, z_output2, bp1_output, bp1_output2, bp2_output, bp2_output2= plotData.plotZ(z_value_frame, analysis_path)
plotData.plotzcase(data, z_value_frame, analysis_path)
plotData.plotQuantile(data, kpi_frame, analysis_path)
# 供应商
print "plot 供应商..."
plotData.plot_supp(z_value_frame[z_value_frame.actual_origin_rate > 0], analysis_path)  # 供应商-sku维度的vlt
table2,manzu_output1,manzu_output2 = plotData.plotsupp2(supp_value_frame, analysis_path)
# 仿真
# sim_path = r'D:\tmp\simulatePrograme\SkuSimulationPbs\2016-11-23_17-45-52\simulation_results'
print "plot 仿真图..."
table3 = plotData.plotsim(sim_path, analysis_path, data.item_sku_id.unique().size - sim_screen.three_conditions.sum())


# ===============================================================================
# =                                 7、输出报告                                  =
# ===============================================================================
print "输出报告..."
# 第一版：输出报告
report = rep.SimReport(name, sort_id, period=','.join(sim_date_range), file_path=analysis_path + '\\report')
report.print_report(stats1)


# 第二版：输出报告
reload(rep)
report_name = str(pinleiid)
period = sim_date_range
file_path = analysis_path + '\\report'
reportword = rep.ReportWord(report_name, period, file_path, data, sim_screen, total_sku, screen_sku)
# 生成报告
reportword.pinleianalysis(pinlei, pinleiid)
reportword.kpianalysis(kpi_cr_str, kpi_ito_str, sim_date_range)
reportword.buhuoanalysis(this_path, z_output, z_output2)
reportword.bpanalysis(bp1_output, bp1_output2, bp2_output, bp2_output2)
reportword.suppanalysis(table2,manzu_output1,manzu_output2)
reportword.simanalysis(table3)
reportword.closeword()
# reportword.generateword(pinlei, pinleiid, kpi_cr_str, kpi_ito_str)
# test the upload author.


