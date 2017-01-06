# encoding:utf-8
__author__ = 'xugang&tanxiao'
import numpy as np
import pandas as pd
import ast
import math
from pandas import DataFrame


# 计算周转天数与现货率
# 能输出每一个sku的，没有过滤
def calcKpi(df, filePath):
    # df = data; filePath = analysis_path
    kpi = DataFrame()
    grouped = df.groupby('item_sku_id')
    kpi['mean_stock'] = grouped.stock_qtty.mean()
    kpi['mean_sale'] = grouped.total_sales.mean()
    kpi['total_days'] = grouped.day_string.count()  # 之后可以和上柜状态联系起来，计算上柜条件下的现货率
    kpi['onsale_days'] = grouped.agg({'stock_qtty': lambda x: x[x > 0].index.size})
    kpi['TD'] = kpi.mean_stock / (kpi.mean_sale + 0.001)  # 克服销量为0时计算为Inf的情况
    kpi['CR'] = kpi.onsale_days / (kpi.total_days)  # 不太明白为什么以前会加0.001，防止全部处于下柜状态？
    # kpi.TD[kpi.TD > 100] = 100
    kpi.to_csv(filePath + '\\kpi.csv')
    return kpi


# 计算Z值，以及补货量（bp），供应商满足率
# 返回的值，包含了所有订货单的情况。其中有些可能是回告没有的，算作无效？
# 采用预测销量来看，不再使用历史28天均值；vlt使用实际的
def calcZ(df, filePath):
    # df = data; filePath = analysis_path
    grouped = df.groupby('item_sku_id')
    summary = {}
    for item_sku_id, group in grouped:
        # item_sku_id = grouped.groups.keys()[1]
        # group = grouped.get_group(item_sku_id)
        # pur_bill_id：采购单编号
        # ==================================================================================================
        sample = group[group.pur_bill_id.isnull() == False]  # 采购单编号还可以为空？
        for i in sample.index:
            # i = sample.index[0]
            # i表示订单号不为空的每日的
            # S（补货点） = 库存 + 采购未入库量 + 内配入库数量 - 销售预定数量 - 内配出库数量
            S = sample.stock_qtty[i] + sample.pur_non_into_wh_qtty[i] + sample.inner_in_qtty[i] - \
                sample.sales_reserve_qtty[i] - sample.inner_outer_qtty[i]
            vlt = sample.vlt[i]
            # supp_brevity_cd：供应商简码
            supp_brevity_cd = sample.supp_brevity_cd[i]
            # supp_name：供应商名称
            supp_name = sample.supp_name[i]
            if np.isnan(vlt):
                # ==========================================================================================
                # 使用 vlt 的均值来填充
                vlt = np.nanmean(df.vlt)  # 有采购单而vlt为空的数目很少
            vlt = int(vlt)
            if vlt > 28:
                # 大于28天的 vlt 限制成28天
                vlt = 28
            if isinstance(sample.ofdsales[i], str):
                # 平均预测销售的均值
                mean_sales = np.mean(ast.literal_eval(sample.ofdsales[i])[:vlt])
            else:
                mean_sales = np.nanmean(group.total_sales.ix[group.index[0]:(i - 1)])  # 没有预测销量的用前面均值代替
            if isinstance(sample.variance[i], str):
                std_list = ast.literal_eval(sample.variance[i])[:vlt]
                sum_std = math.sqrt(np.sum([x ** 2 for x in std_list]))
            else:
                sum_std = np.nanstd(group.total_sales.ix[group.index[0]:(i - 1)]) * math.sqrt(vlt)
            z_value = (S - mean_sales * vlt) / (sum_std + 0.001)
            # ==============================================================================================
            # group.loc[:,["originalnum","actual_pur_qtty","plan_pur_qtty"]] 好晕。
            # originalnum：原始下单数量/平均预测销量？
            bp = sample.originalnum[i] / (mean_sales + 0.001)
            # actual_pur_qtty：实际采购数量
            # plan_pur_qtty：计划采购数量
            # actual_plan_rate = 实际/计划
            actual_plan_rate = sample.actual_pur_qtty[i] / float(sample.plan_pur_qtty[i])  # 这个计算出来为空的，应该是订单无效？
            # actual_origin_rate = 实际/原始
            actual_origin_rate = sample.actual_pur_qtty[i] / float(sample.originalnum[i])
            summary[i] = {'item_sku_id': item_sku_id, 'day_string': sample.day_string[i],
                          'band': sample.org_nation_sale_num_band[i],
                          'z_value': z_value, 'bp': bp, 'vlt': vlt, 'supp_brevity_cd': supp_brevity_cd,
                          'supp_name': supp_name,
                          'actual_plan_rate': actual_plan_rate, 'actual_origin_rate': actual_origin_rate}
    z_value_frame = DataFrame.from_dict(summary).T
    # z值限制在 -10~100 中
    z_value_frame.z_value = z_value_frame.z_value.apply(lambda x: 100 if x > 100 else x)
    z_value_frame.z_value = z_value_frame.z_value.apply(lambda x: -10 if x < -10 else x)
    z_value_frame.bp = z_value_frame.bp.apply(lambda x: 100 if x > 100 else x)
    z_value_frame.to_csv(filePath + '\\z_value_frame.csv', index=False)
    return z_value_frame


def calcsupp(df, filePath):
    '''
    计算供应商的vltcv、满足率（actual_origin_rate）、相对满足率
    :param df:
    :param filePath:
    :return:
    '''
    # df = data; filePath = analysis_path
    grouped = df.groupby('supp_name')
    summary = {}
    for supp_name, group in grouped:
        # item_sku_id = grouped.groups.keys()[1]
        # group = grouped.get_group(item_sku_id)
        sample = group[group.pur_bill_id.isnull() == False]
        for i in sample.index:
            # i = sample.index[0]
            vlt = sample.vlt[i]
            # supp_brevity_cd：供应商简码
            supp_brevity_cd = sample.supp_brevity_cd[i]
            if np.isnan(vlt):
                # 使用 vlt 的均值来填充
                vlt = np.nanmean(df.vlt)  # 有采购单而vlt为空的数目很少
            vlt = int(vlt)
            if vlt > 28:
                # 大于28天的 vlt 限制成28天
                vlt = 28
            actual_plan_rate = sample.actual_pur_qtty[i] / float(sample.plan_pur_qtty[i])  # 这个计算出来为空的，应该是订单无效？
            # actual_origin_rate = 实际/原始
            actual_origin_rate = sample.actual_pur_qtty[i] / float(sample.originalnum[i])
            summary[i] = {'supp_name': supp_name, 'day_string': sample.day_string[i],
                          'band': sample.org_nation_sale_num_band[i], 'item_sku_id': sample.item_sku_id[i],
                          'vlt': vlt, 'supp_brevity_cd': supp_brevity_cd,
                          'actual_pur_qtty': sample.actual_pur_qtty[i],'plan_pur_qtty': sample.plan_pur_qtty[i],
                          'originalnum': sample.originalnum[i],
                          'pur_bill_id': sample.pur_bill_id[i],
                          'actual_plan_rate': actual_plan_rate, 'actual_origin_rate': actual_origin_rate}
    z_value_frame = DataFrame.from_dict(summary).T
    z_value_frame.to_csv(filePath + '\\supp_value_frame.csv', index=False)
    return z_value_frame



