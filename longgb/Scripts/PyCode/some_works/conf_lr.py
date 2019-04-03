# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import Imputer
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.types import *
from pyspark import StorageLevel
from statsmodels.regression.quantile_regression import QuantReg
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from tsfresh.utilities.dataframe_functions import impute
import tsfresh.feature_extraction.feature_calculators as feature_calc
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import re
import os
import os.path
import sys
import math
from datetime import datetime, timedelta
import shutil
import random
from fbprophet import Prophet
# from tsfresh_param import *
import lightgbm as lgb

quantile_regression_on = True
weather_as_features = False
# promotion_on = False
promotion_on = True

data_key_columns_map_sales = \
    {'salesForecast': 'sale_num', \
     'priceBeforeDiscount': 'avg_jd_unit_price', \
     'priceAfterDiscount': 'avg_sale_unit_price', \
     'stockQuantity': 'stock_qtty', \
     'vendibility': 'vendibility', \
     'sales': 'sale_num', \
     'salesAll': 'sale_num'}

data_key_columns_map_promotion = \
    {'maxsyntheticdiscounta': 'synthetic_discount', \
     'promotiontype': 'promotion_type', \
     'promotionsubtype': 'promotion_sub_type'}

data_key_columns_map = dict(list(data_key_columns_map_sales.items()) \
                            + list(data_key_columns_map_promotion.items()))

# added 2018/12/29
# 春节调整部分函数使用的一级品类
spring_festivals_1_cates = ['1320', '12259']


def get_promotion_info(sc, path='/user/cmo_ipc/sfs_ai/lr/lr_factors/sfs_sku_promotions_timeseries/'):
    if not promotion_on:
        return sc.parallelize([('0', None)])
    promo_rdd_list = []
    promo_timeseries_columns = ['tag', 'identity', 'data_key', 'start_date', 'len', 'time_interval', 'x', 'y', 'key']
    for val in data_key_columns_map_promotion.keys():
        rdd_tmp = sc.textFile(path + 'key={0}'.format(val)) \
            .map(lambda line: line.split('\001')) \
            .map(lambda line: (line[promo_timeseries_columns.index('identity')], \
                               [line[promo_timeseries_columns.index(col)] for col in timeseries_filtered_columns])) \
            .coalesce(1000)
        promo_rdd_list.append(rdd_tmp)
    promo_rdd = sc.union(promo_rdd_list) \
        .groupByKey() \
        .filter(lambda line: len(line[1]) == len(data_key_columns_map_promotion.items())) \
        .map(lambda line: (line[0].split('#')[-1], line[1])) \
        .coalesce(1000)
    return promo_rdd


big_promo_618 = pd.DataFrame({
    'holiday': 'big_promo_618', \
    'ds': pd.to_datetime(['2015-06-18', '2016-06-18', '2017-06-18', '2018-06-18', '2019-06-18']), \
    'lower_window': -18, \
    'upper_window': 3, \
    })

big_promo_1111 = pd.DataFrame({
    'holiday': 'big_promo_1111', \
    'ds': pd.to_datetime(['2015-11-11', '2016-11-11', '2017-11-11', '2018-11-11', '2019-11-11']), \
    'lower_window': -11, \
    'upper_window': 3, \
    })

big_promo_1212 = pd.DataFrame({
    'holiday': 'big_promo_1212', \
    'ds': pd.to_datetime(['2015-12-12', '2016-12-12', '2017-12-12', '2018-12-12', '2019-12-12']), \
    'lower_window': -1, \
    'upper_window': 1, \
    })

big_promo_815 = pd.DataFrame({
    'holiday': 'big_promo_815', \
    'ds': pd.to_datetime(['2015-08-15', '2016-08-15', '2017-08-15', '2018-08-15', '2019-08-15']), \
    'lower_window': -2, \
    'upper_window': 2, \
    })

spring_festivals = pd.DataFrame({
    'holiday': 'spring_festival_2', \
    'ds': pd.to_datetime(['2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05']), \
    'lower_window': -7, \
    'upper_window': 7, \
    })

new_years = pd.DataFrame({
    'holiday': 'new_year_1', \
    'ds': pd.to_datetime(['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']), \
    'lower_window': -7, \
    'upper_window': 2, \
    })

women_days = pd.DataFrame({
    'holiday': 'women_day_38', \
    'ds': pd.to_datetime(['2015-03-08', '2016-03-08', '2017-03-08', '2018-03-08', '2019-03-08']), \
    'lower_window': -7, \
    'upper_window': 2, \
    })

school_days = pd.DataFrame({
    'holiday': 'school_day_91', \
    'ds': pd.to_datetime(['2015-09-01', '2016-09-01', '2017-09-01', '2018-09-01', '2019-09-01']), \
    'lower_window': -14, \
    'upper_window': 2, \
    })

children_days = pd.DataFrame({
    'holiday': 'children_day_61', \
    'ds': pd.to_datetime(['2015-06-01', '2016-06-01', '2017-06-01', '2018-06-01', '2019-06-01']), \
    'lower_window': -7, \
    'upper_window': 1, \
    })


def flatten_holiday():
    df_holiday = pd.concat([big_promo_618, big_promo_1111, \
                            big_promo_1212, big_promo_815, \
                            new_years, spring_festivals,
                            women_days, school_days, children_days])
    holidays = []
    for holiday_type, row in df_holiday.iterrows():
        for i in range(row['lower_window'], row['upper_window'] + 1):
            dt = row['ds'] + timedelta(i)
            holidays.append([int(float(row['holiday'].split('_')[-1])), dt])
    df = pd.DataFrame(holidays, columns=['holiday', 'order_date'])
    return df.groupby('order_date').min().reset_index()


def get_holidays(keys):
    holidays = pd.concat([big_promo_618, big_promo_1111, \
                          big_promo_1212, big_promo_815, \
                          new_years, spring_festivals,
                          women_days, school_days])
    return holidays


def get_data_df(rdd, columns):
    data = rdd.collect()
    df = pd.DataFrame(data, columns=columns)
    return df


def get_history_weather(sc, dc_id, s_str, e_str, columns, dc_type=[0]):
    sql = """select dc_id, v_date as order_date,
              daytemperaturehigh as tmp_high,
              daytemperaturelow as tmp_low 
              from 
              (select DISTINCT dc_id, dc_name from dim.dim_dc_info) a join 
                (select * from fdm.fdm_cis_products_weatherforecast_chain where dp='ACTIVE'
                  and v_date >= '{0}' and v_date <= '{1}') b 
              on a.dc_name = b.cityname""".format(s_str, e_str)
    df = get_data_df(get_data_rdd(sc, sql, columns), columns)
    df.loc[:, 'order_date'] = pd.to_datetime(df['order_date'])
    df_copy = df.copy(deep=True)
    df_copy.loc[:, 'order_date'] = df_copy['order_date'].map(lambda x: x + timedelta(365))
    for col in weather_columns[2:]:
        df_copy.rename(columns={col: col + '_yoy'}, inplace=True)
    df = pd.merge(df_copy, df, left_on=['dc_id', 'order_date'], right_on=['dc_id', 'order_date'], how='left')
    for col in weather_columns[2:]:
        df[col].fillna(df[col + '_yoy'], inplace=True)
    return df[weather_columns]


weather_columns = ['dc_id', 'order_date', 'tmp_high', 'tmp_low']
sep = ','
period = 7
predict_periods_num = 13
cate_id_list = [652, 670, 737, 1315, 1316, 1318, 1319, 1320, 1620, 1672, 5025, 6144, 6196, 6233, 6728, 6994, 9192, 9847,
                9855, 9987, 11729, 12218, 12259]
dc_id_list = [3, 4, 5, 6, 9, 10, 316, 682, 772]

item_skus_columns = ['item_sku_id', 'item_first_cate_cd', \
                     'item_second_cate_cd', 'item_third_cate_cd', 'brand_code']

category_keys_columns = item_skus_columns[1:-1]
keys_columns = ['item_sku_id', 'dc_id'] + item_skus_columns[1:]


def get_sql(table_name, cond=None):
    sql = """select {0} from {1}""".format('*', table_name)
    if not (cond is None):
        sql = '{0} where {1}'.format(sql, cond)
    return sql


def get_data_rdd(sc, sql, columns):
    hc = HiveContext(sc)
    hc.setConf("hive.exec.orc.split.strategy", "BI")
    results = hc.sql(sql) \
        .rdd \
        .map(lambda x: [x[col] for col in columns])
    return results


calendar_columns = ['f_key', 'f_g_week_of_year', 'f_l_solar']


def get_calendar_info(sc, start_dt_str, end_dt_str):
    cols = ','.join(calendar_columns)
    sql = "select {0} from app.app_sfs_calendar where f_key >= '{1}' and f_key <= '{2}'" \
        .format(cols, start_dt_str, end_dt_str)
    rdd = get_data_rdd(sc, sql, calendar_columns).collect()
    df = pd.DataFrame(rdd, columns=calendar_columns)
    df.rename(columns={'f_key': 'order_date', 'f_g_week_of_year': 'solar_week', 'f_l_solar': 'lunar_week'},
              inplace=True)
    # df.loc[df['lunar_week'] == 'NULL','lunar_week'] = np.NaN
    df['lunar_week'].fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    df['order_date'] = pd.to_datetime(df['order_date'])
    if weather_as_features:
        weather = get_history_weather(sc, 0, '2013-01-01', '2020-01-01', weather_columns)
        weather.loc[:, 'tmp_high'] = pd.to_numeric(weather['tmp_high'])
        weather.loc[:, 'dc_id'] = pd.to_numeric(weather['dc_id'])
        weather_national = weather[['order_date', 'dc_id', 'tmp_high']].groupby('order_date').mean().reset_index()
        weather_national.loc[:, 'dc_id'] = 0
        weather = pd.concat([weather[['order_date', 'dc_id', 'tmp_high']], weather_national])
        df = pd.merge(df, weather, left_on='order_date', right_on='order_date', how='left')
        # df['dc_id'].fillna(method='ffill',inplace=True)
        df['tmp_high'].fillna(method='ffill', inplace=True)
    return df


def get_dc_list(sc, dc_type=[0]):
    sql = 'select distinct dc_id from dim.dim_dc_info where dc_type in ({0})' \
        .format(','.join([str(dc) for dc in dc_type]))
    dc = get_data_rdd(sc, sql, ['dc_id']) \
        .map(lambda x: x[0]) \
        .collect()
    return dc


def get_cate_info(sc, dt_str, ratio_path=None, book=True, filter_col=None, cate_id=None):
    if book:
        gdm_m03_sql = \
            get_sql('gdm.gdm_m03_item_sku_da', \
                    cond=""" dt = '{0}' and sku_valid_flag = 1 \
                              and (data_type in (1,2) or (data_type=3 \
                              and pop_coop_mode_cd =1))""".format(dt_str))
    else:
        gdm_m03_sql = \
            get_sql('gdm.gdm_m03_self_item_sku_da', \
                    cond="""dt = '{0}' and sku_valid_flag = 1""".format(dt_str))
    if not (filter_col is None):
        gdm_m03_sql += ' and {0}={1}'.format(filter_col, cate_id)
    item_skus_cate = get_data_rdd(sc, gdm_m03_sql, item_skus_columns)
    if ratio_path is None:
        item_skus_cate = item_skus_cate.map(lambda x: (str(x[item_skus_columns.index('item_sku_id')]), \
                                                       ('#'.join([str(x[item_skus_columns.index(c)]) for c in
                                                                  item_skus_columns[1:]]), None)))
    else:
        ratio_info = get_ratio_info(sc, ratio_path)
        item_skus_cate = item_skus_cate.map(lambda x: ('#'.join([str(x[item_skus_columns.index('item_third_cate_cd')]), \
                                                                 str(x[item_skus_columns.index('brand_code')])]), [x])) \
            .reduceByKey(lambda x, y: x + y) \
            .leftOuterJoin(ratio_info) \
            .flatMap(lambda y: [(str(x[item_skus_columns.index('item_sku_id')]), \
                                 ('#'.join([str(x[item_skus_columns.index(c)]) for c in item_skus_columns[1:]]), \
                                  y[1][1])) for x in y[1][0]])

        ratio_info.unpersist()
    return item_skus_cate


def robust_op(ll, op=np.nanmean, default=0.0):
    r = op(ll)
    return (default if np.isnan(r) else r)


def str_to_datetime(d_str):
    return datetime.strptime(d_str[:10], '%Y-%m-%d')


order_features = ['sku_id', 'dc_id', 'order_date', \
                  'sale_num', 'stock_qtty', 'avg_sale_unit_price', 'avg_jd_unit_price']


def feature_index(f_str):
    return order_features.index(f_str)


timeseries_columns = ['tag', 'identity', 'data_key', 'start_date', 'len', 'time_interval', 'y', 'key']
timeseries_filtered_columns = ['identity', 'data_key', 'start_date', 'len', 'y']


def timeseries_index(c_str):
    return timeseries_columns.index(c_str)


def timeseries_filtered_index(c_str):
    return timeseries_filtered_columns.index(c_str)


sfs_rdc_forecast_columns = ['type', 'sku_id', 'dc_id', 'fdsales', 'fwsales', 'ofdsales', 'ofwsales', 'dt']


def sfs_rdc_forecast_index(f_str):
    return sfs_rdc_forecast_columns.index(f_str)


sfs_rdc_forecast_original_index = [0, 1, 2, 7, 8, 10, 11, -1]


def get_sfs_rdc_forecast_result(sc, date_str):
    rdc_forecast = sc.textFile('{0}/app.db/app_sfs_rdc_forecast_result/dt={1}' \
                               .format('/user/cmo_ipc/', date_str)) \
        .map(lambda line: line.split('\001')) \
        .map(lambda line: line + [date_str]) \
        .map(lambda line: [str(line[i]) for i in sfs_rdc_forecast_original_index])
    return rdc_forecast


def get_online_forecast_result(sc, dt_str, days_num):
    start_dt = datetime.strptime(dt_str, '%Y-%m-%d')
    dt_str_list = [dt.strftime('%Y-%m-%d') for dt in pd.date_range(start_dt, start_dt + timedelta(days_num))]

    forecast_result = get_sfs_rdc_forecast_result(sc, dt_str) \
        .flatMap(lambda line: [[str(line[sfs_rdc_forecast_index('sku_id')]), \
                                str(line[sfs_rdc_forecast_index('dc_id')]), \
                                dt, order_num] \
                               for dt, order_num in \
                               zip(dt_str_list, \
                                   eval(line[sfs_rdc_forecast_index('ofdsales')])[:days_num])])
    return forecast_result


filter_order_mean_threshold = 0.05


def filter_sku(record):
    sale_dict = {}
    for item in record[1]:
        stock_qtty = float(float(item[order_promotion_index('stock_qtty')]))
        order_num = float(float(item[order_promotion_index('order_num')]))
        if stock_qtty == 0 and order_num == 0:
            continue
        key = str(item[order_promotion_index('dt')])[:10]
        sale_num = int(float(item[order_promotion_index('sale_qtty')]) \
                       * float(item[order_promotion_index('order_num')]))
        if sale_dict.has_key(key):
            sale_dict[key] = sale_dict[key] + sale_num
        else:
            sale_dict[key] = sale_num
    sale_mean = 0.0
    if bool(sale_dict):
        sale_mean = robust_op(sale_dict.values())
    if sale_mean < filter_order_mean_threshold:
        return []
    else:
        return record[1]


def compute_mapd(reals, predicts):
    total_reals = robust_op(reals, op=np.nansum)
    total_abs_diff = robust_op(np.abs(reals - predicts), op=np.nansum)
    return total_abs_diff if total_reals == 0.0 else total_abs_diff / total_reals


records_num_threshold = 30
history_periods = 3


def weighted_avg(sales_list, forecast_period_span, forecast_period):
    forecast_span = forecast_period_span * forecast_period
    sales_upper_bound = np.percentile(np.unique(sales_list), 100)
    sales_clean = sales_list[sales_list <= sales_upper_bound][-forecast_span:]
    # print('weighted_avg')
    # print(sales_list)
    results = [0.0] * forecast_period_span
    records_low_bound = 14
    if len(sales_clean) < records_low_bound:
        sales_clean = np.append([0.0] * (records_low_bound - len(sales_clean)), sales_clean)
    for i in range(1, forecast_span + 1):
        total_weight = 0.0
        weighted_sale = 0.0
        sales_sample = []
        for sale in sales_clean:
            if random.random() < 0.4:
                sales_sample.append(sale)
        for j, sale in enumerate(sales_sample):
            weight = j
            total_weight += weight
            weighted_sale += float(sale * weight)
        # sale = (0.0 if total_weight == 0.0 else weighted_sale / total_weight)
        # sales_clean = np.append(sales_clean,[sale])
        sale = robust_op(sales_sample)
        results[(i - 1) // forecast_period] += sale
    return results


def linear_regression(LR, features, targets, alpha, intercept=False):
    rr = LR(alpha=alpha, fit_intercept=intercept, normalize=True)
    rr.fit(features, targets)
    r2 = rr.score(features, targets)
    targets_hat = np.array(rr.predict(features))
    coefs = np.array(rr.coef_)
    return targets_hat, r2, coefs, rr.intercept_, rr


def quantile_regression(train_features, \
                        train_targets, \
                        forecast_span, \
                        period_sales, \
                        seasonal, \
                        latest_periods_index, \
                        seasonal_periods_index, \
                        qs=np.arange(.5, .7, .01)):
    data = [train_features[i] + [t] for i, t in enumerate(train_targets)]
    df = pd.DataFrame(data, columns=['feature%i' % i for i in range(len(train_features[0]))] + ['target'])
    formula = 'target ~ '
    result_columns = ['q', 'intercept']
    for i in range(len(train_features[0])):
        if i == 0:
            formula += 'feature%i' % i
        else:
            formula += '+feature%i' % i
        result_columns.append('coef%i' % i)
        result_columns.append('coefl%i' % i)
        result_columns.append('coefu%i' % i)
    result_columns.append('r2')
    result_columns.append('mapd')
    mod = smf.quantreg(formula, df)
    results = []
    mapd_compute_records = 4
    for q in qs:
        try:
            res = mod.fit(q=q)
            result = [q, res.params['Intercept']]
            weights = []
            for i in range(len(train_features[0])):
                weights.append(res.params['feature%i' % i])
                result.append(res.params['feature%i' % i])
                result.extend(res.conf_int().ix['feature%i' % i].tolist())
            result.append(res.rsquared)
            targets_hat = []
            targets = []
            for _, row in df.iterrows():
                targets.append(row['target'])
                fr = np.dot(row[:-1], weights) + res.params['Intercept']
                targets_hat.append(fr)
            mapd = compute_mapd(np.array(targets[-mapd_compute_records:]), \
                                np.array(targets_hat[-mapd_compute_records:]))
            result.append(mapd)
        except:
            result = [np.NaN] * (len(train_features[0]) * 3 + 4)
        results.append(result)
    df_res = pd.DataFrame(results, columns=result_columns)
    predict_results_all = []
    for _, row in df_res.iterrows():
        predict_results = []
        features = period_sales[latest_periods_index + seasonal_periods_index]
        features = np.array(features)
        weights = row[['coef%i' % j for j in range(len(train_features[0]))] + ['intercept']]
        for i in range(forecast_span):
            result = np.dot(weights, np.append(features, [1]))
            result = (result if np.isnan(result) else max(0.0, result))
            predict_results.append(result)
            if seasonal:
                his_predict_sales = np.append(period_sales, predict_results)
                features = np.append(his_predict_sales[latest_periods_index], \
                                     his_predict_sales[seasonal_periods_index])
            else:
                features = np.roll(features, 1)
                features[0] = result
        predict_results_all.extend(predict_results)
    predict_results_all = np.array(predict_results_all).reshape(df_res.shape[0], forecast_span).T
    for i in range(forecast_span):
        df_res['forecast%i' % i] = predict_results_all[i]
    return df_res


def ransac_regression(LR, features, targets, alpha, intercept=False):
    lr = LR(alpha=alpha, fit_intercept=intercept, normalize=True)
    rr = linear_model.RANSACRegressor(lr)
    rr.fit(features, targets)
    r2 = rr.score(features, targets)
    targets_hat = np.array(rr.predict(features))
    coefs = np.array(rr.estimator_.coef_)
    return targets_hat, r2, coefs, rr.estimator_.intercept_, rr


def order_clean_by_price(df, debug):
    if df.shape[0] <= records_num_threshold:
        return df
    org_columns = df.columns
    df['avg_jd_unit_price'] = np.where(df['avg_jd_unit_price'] < 0.0, 0.0, df['avg_jd_unit_price'])
    df['avg_sale_unit_price'] = np.where(df['avg_sale_unit_price'] < 0.0, 0.0, df['avg_sale_unit_price'])
    df['jd_price_log'] = np.log(df['avg_jd_unit_price'] + 1)
    df['sale_price_log'] = np.log(df['avg_sale_unit_price'] + 1)
    df['sale_num_log'] = np.log(df['sale_num'] + 1)
    A = np.vstack([df['sale_price_log'].tolist()]).T
    r2 = np.NaN
    weights = np.NaN
    intercept = np.NaN
    regressor = np.NaN
    for lr_method in [Ridge, Lasso]:
        for alpha in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                targets_hat, cur_r2, cur_weights, cur_intercept, cur_regressor \
                    = linear_regression(lr_method, A, df['sale_num_log'].tolist(), alpha, intercept=True)
                if r2 is np.NaN or cur_r2 > r2:
                    r2 = cur_r2
                    weights = cur_weights
                    intercept = cur_intercept
                    regressor = cur_regressor
            except:
                continue
    price_log_mean = df['sale_price_log'].tail(21).mean()
    debug_print('price_mean: {0}, r2: {1}'.format(np.exp(price_log_mean) - 1, r2), debug)
    if r2 >= 0.5:
        a = weights[0]
        df['price_diff'] = df['sale_price_log'].tail(21).mean() - df['sale_price_log']
        df['sale_num'] = df['sale_num'] * np.exp(a * df['price_diff'])
    return df[org_columns]


def try_parse(s):
    try:
        f = float(s)
        return f
    except:
        return np.NaN


def debug_print(s, debug):
    if debug:
        print(s)


def harmonious_mean(ll):
    zero_count = 0
    non_zero_count = 0
    weight = 0.0
    for l in ll:
        if l > 0.0:
            weight += 1.0 / l
            non_zero_count = non_zero_count + 1
        else:
            zero_count = zero_count + 1
    adj = (non_zero_count / weight if weight > 0.0 else 0.0)
    return ((adj * non_zero_count) / float(len(ll)) if len(ll) > 0.0 else 0.0)


def remove_outliers(ls, ratio=2.0):
    ls_mean = robust_op(ls)
    ls_std = np.std(ls)
    new_ls = []
    for i in ls:
        if i >= ls_mean + ratio * ls_std:
            new_ls.append(ls_mean + ls_std)
        elif i <= ls_mean - ratio * ls_std:
            new_ls.append(max(0.0, ls_mean - ls_std))
        else:
            new_ls.append(i)
    return np.array(new_ls)


recent_day_num = 28


# To check whether JD price is changed permanently
# If yes and sales are affected remarkably,
# time series corresponding to old JD prices will be truncated
def jd_price_clean(df, predict_cur_date, debug=False):
    recent_jd_price_mean = df.loc[df.order_date >= predict_cur_date - timedelta(7), \
                                  'avg_jd_unit_price'].mean()
    jd_price_sale_list = df.loc[(df.order_date >= predict_cur_date - timedelta(56)) \
                                & (df.avg_jd_unit_price <= 1.2 * recent_jd_price_mean) \
                                & (df.avg_jd_unit_price >= 0.8 * recent_jd_price_mean) \
                                & ((df.sale_num > 0) | (df.stock_qtty > 0)), 'sale_num']
    debug_print('jd_price_sale_list', debug)
    debug_print(jd_price_sale_list, debug)
    debug_print('recent_jd_price: {0}'.format(recent_jd_price_mean), debug)
    if jd_price_sale_list.shape[0] <= 7:
        return df  # df.loc[df.order_date >= predict_cur_date - timedelta(21), :]
    jd_price_sale_list = baseline_smooth(list(jd_price_sale_list), step=28)
    sale_mean = np.nanmean(jd_price_sale_list)
    df.loc[(df.avg_jd_unit_price > 1.2 * recent_jd_price_mean), 'sale_num'] \
        = df.loc[(df.avg_jd_unit_price > 1.2 * recent_jd_price_mean), 'sale_num'].map(lambda x: max(x, sale_mean))
    df.loc[(df.avg_jd_unit_price < 0.8 * recent_jd_price_mean), 'sale_num'] \
        = df.loc[(df.avg_jd_unit_price < 0.8 * recent_jd_price_mean), 'sale_num'].map(lambda x: min(x, sale_mean))
    debug_print('recent_jd_price: {0}, jd_price_sale_mean: {1}'.format(recent_jd_price_mean, sale_mean), debug)
    return df


def forecast_result_ensemble(keys, df_agg, df_promotion, predict_cur_date, \
                             debug=False, jd_price_adjusted=False, \
                             ratio_info=None, calendar=None):
    jd_price_adjusted = apply_price_adjusted(keys)
    debug_print('price adjusted: {0}'.format(jd_price_adjusted), debug)
    if jd_price_adjusted:
        df_agg = jd_price_clean(df_agg, predict_cur_date, debug)
    return sale_forecast_lr(keys, df_agg, df_promotion, predict_cur_date, debug, \
                            jd_price_adjusted, \
                            ratio_info, calendar)


big_promotion_list = ['06-18', '11-11']


def apply_seasonal(keys):
    return True
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_first_cate_cd')] \
                in ['12218', '1318', '1620', '1315', '652', '670', '737', '1316', '1320']) \
               | (keys[keys_columns.index('item_second_cate_cd')] in []))
    except:
        ans = False
    return ans


def apply_seasonal_overwrite(keys):
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_first_cate_cd')] \
                in []) \
               | (keys[keys_columns.index('item_second_cate_cd')] in []))
    except:
        ans = False
    return ans


# control whether and which categories should tsfresh be applied
def apply_tsfresh(keys):
    return True
    ans = False
    cate1_list = ['12218', '652', '670', \
                  '737', '1318', '15248', '1316', \
                  '1320', '6994', '9855', \
                  '15901', '16750', '12259', \
                  '6196', '1319', '1713', '6728', \
                  '1672', '9847', '1620', '9987', \
                  '9192']
    # cate1_list = ['652','670','15248','1316','6994','9855','1320','737','12218','1319','6728']
    try:
        ans = (((keys[keys_columns.index('item_first_cate_cd')] \
                 in cate1_list) \
                | (keys[keys_columns.index('item_second_cate_cd')] in [])) \
               & (keys[keys_columns.index('dc_id')] in ['3', '4', '5', '6', '9', '10', '316', '772']))
    except:
        ans = False
    return ans


def tsfresh_sale_threshold(keys):
    thd = 20.0
    try:
        if ((keys[keys_columns.index('item_first_cate_cd')] \
             in ['1316', '1320']) \
                | (keys[keys_columns.index('item_second_cate_cd')] in [])):
            thd = 50.0
        elif (keys[keys_columns.index('item_first_cate_cd')] in ['737']):
            thd = 5.0
    except:
        pass
    return thd


# control whether and which categories should fbprobet be applied
def apply_fbprophet(keys):
    return False
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_first_cate_cd')] in ['670']) \
               | (keys[keys_columns.index('item_second_cate_cd')] in []))
    except:
        ans = False
    return ans


# control which categories should big promotion tuning be ignored
def apply_big_promotion(keys):
    ans = True
    try:
        ans = not (keys[keys_columns.index('item_second_cate_cd')] in ['830'])
    except:
        ans = True
    return ans


# control init cut pos by categories, the bigger of cut pos,
# the less exceptionally high sales will be elimindated
def get_init_cut_pos(keys):
    try:
        if keys[keys_columns.index('item_first_cate_cd')] in ['670', '737']:
            return 85
        elif keys[keys_columns.index('item_second_cate_cd')] in ['830']:
            return 80
        elif keys[keys_columns.index('item_first_cate_cd')] in ['12218']:
            return 80
        elif keys[keys_columns.index('item_third_cate_cd')] in ['747']:
            return 90
        else:
            return 75
    except:
        return 75


def get_out_of_stock_threshold(keys):
    t = 42
    try:
        if (keys[keys_columns.index('item_first_cate_cd')] in \
            ['1320', '12259', '16750', '1319', '1316', '6233', '12473', '15901', '6994']) \
                or (keys[keys_columns.index('item_third_cate_cd')] in ['878', '880', '14420', '12392', '12401']):
            t = 91
    except:
        t = 42
    return t


def apply_price_adjusted(keys):
    # return True
    ans = False
    # print('keys: {0}'.format(keys))
    try:
        ans = ((keys[keys_columns.index('item_second_cate_cd')] in ['0', '830']) \
               | (keys[keys_columns.index('item_first_cate_cd')] in ['12218', '9987']))
    except:
        ans = False
    return ans


def apply_ratio(keys):
    ans = True
    try:
        ans = (not (keys[keys_columns.index('item_first_cate_cd')] in []))
    except:
        ans = True
    return ans


def apply_vendibility_smooth(keys):
    ans = True
    try:
        ans = not ((keys[keys_columns.index('item_second_cate_cd')] in []) \
                   | (keys[keys_columns.index('item_first_cate_cd')] in []))
    except:
        ans = True
    return ans


def seasonal_upper_threshold(keys):
    try:
        if ((keys[keys_columns.index('item_third_cate_cd')] in ['870']) \
                | (keys[keys_columns.index('item_second_cate_cd')] in []) \
                | (keys[keys_columns.index('item_first_cate_cd')] in ['12218'])):
            return 1.5
    except:
        pass
    return 1.5


def seasonal_lower_threshold(keys):
    try:
        if ((keys[keys_columns.index('item_third_cate_cd')] in []) \
                | (keys[keys_columns.index('item_second_cate_cd')] in []) \
                | (keys[keys_columns.index('item_first_cate_cd')] in ['12218'])):
            return 0.5
    except:
        pass
    return 0.9


def tune_results_by_promotion(keys, \
                              df_agg, \
                              df_promotion, \
                              predict_results_day, \
                              predict_cur_date, \
                              debug):
    if df_promotion is None:
        return predict_results_day
    ##promotion tuning is not applied on long tail SKUs
    if harmonious_mean(predict_results_day) <= 0.5:
        return predict_results_day
    debug_print('predict before promotion tunning', debug)
    debug_print(predict_results_day, debug)
    discount_key = 'synthetic_discount'
    df_promotion['discount'] = df_promotion[discount_key] \
        .map(lambda x: (0 if x >= 1.0 else (10 if x <= 0.0 else (10 - int(round(x * 10, 0))))))
    df_sales_promotion = pd.merge(df_agg, df_promotion, left_on='order_date', right_on='order_date', how='left')
    df_sales_promotion['discount'].fillna(10, inplace=True)

    discount_sales = dict(df_sales_promotion[['discount', 'original_sale_num']] \
                          .groupby('discount') \
                          .agg({'original_sale_num': lambda x: np.nanmean(x)}) \
                          .reset_index().values.tolist())
    future_discount = df_promotion.loc[df_promotion['order_date'] >= predict_cur_date, 'discount'].tolist()
    debug_print('future discount: {0}'.format(future_discount), debug)
    debug_print('discount_sales: {0}'.format(discount_sales), debug)
    # baseline_sale = discount_sales.get(10,harmonious_mean(predict_results_day))
    # if baseline_sale <= 0.0:
    max_promotion_threshold = 28
    baseline_sale = harmonious_mean(predict_results_day[:max_promotion_threshold])
    if baseline_sale > 0.0:
        for i, discount in enumerate(future_discount):
            if (discount < 10) and (i < max_promotion_threshold):
                sale = discount_sales.get(discount, predict_results_day[i] * (1.0 + max(0.5, 1.0 - discount / 10.0)))
                promo_ratio = min(5.0, max(1.0, sale / baseline_sale))
                predict_results_day[i] = predict_results_day[i] * promo_ratio
        debug_print('predict after promotion tunning', debug)
        debug_print(predict_results_day, debug)
    return predict_results_day


def normalize_map(mp):
    if len(mp.items()) == 0:
        return mp
    total = robust_op(mp.values(), op=np.nansum)
    if total == 0.0:
        return mp
    for key in mp.keys():
        mp[key] = mp[key] / total
    return mp


def is_seasonal_category(keys, ratio_info):
    ans = False
    if ratio_info is None:
        return ans
    solar_ratio_map = {}
    dc_str = str(keys[keys_columns.index('dc_id')])
    for _, row in ratio_info.iterrows():
        if row['key_name'] in ['solar_' + dc_str, 'solar_' + dc_str + '.0']:
            solar_ratio_map[int(float(row['week']))] = float(row['ratio'])
    if len(solar_ratio_map.values()) >= 40:
        ratio_list = np.array(list(solar_ratio_map.values()))
        ratio_sum = np.sum(ratio_list)
        ratio_percentile = np.percentile(ratio_list, 70)
        ratio_peak_sum = np.sum(ratio_list[ratio_list > ratio_percentile])
        if ratio_sum > 0.0 and ratio_peak_sum / ratio_sum >= 0.5:
            ans = True
    return ans


def tune_results_by_quanreg(keys, predict_results, df_quanreg):
    if df_quanreg is None:
        return predict_results
    min_mapd = df_quanreg.mapd.min()
    for i in range(len(predict_results) - 2):
        quan_result = df_quanreg.loc[(df_quanreg.mapd <= min_mapd + 0.001), ['forecast%i' % i]].mean().iloc[0]
        if np.isnan(quan_result):
            continue
        predict_results[i] = harmonious_mean([predict_results[i], quan_result])
    return predict_results


def smooth_dict(mp, ratio=2.0):
    vals = baseline_smooth_frag(list(mp.values()), absolute_gap=(0, 0), relative_gap=(0.1, 2.5))
    val_mean = robust_op(vals)
    val_std = robust_op(vals, np.nanstd)
    for key in mp.keys():
        val = mp.get(key, np.NaN)
        if np.isnan(val):
            mp[key] = val_mean
        elif (val < val_mean - ratio * val_std):
            #mp[key] = max(0.0, val_mean + ratio * val_std)
            mp[key] = max(0.0, val_mean - ratio * val_std)#2019.1.23 modify + change to -
        elif (val > val_mean + ratio * val_std):
            mp[key] = val_mean + ratio * val_std
    return mp


def tune_results_by_ratio(keys, df_agg, predict_results_day, \
                          predict_cur_date, ratio_info, calendar, debug):
    debug_print('predict before tunning', debug)
    debug_print(predict_results_day, debug)
    debug_print('keys: {0}, ratio_info: {1}, calendar: {2}'.format(keys, ratio_info, calendar), debug)
    if (not apply_ratio(keys)) or (ratio_info is None) or (calendar is None):
        return predict_results_day
    debug_print('tuning results', debug)
    span = len(predict_results_day)
    df = pd.merge(df_agg, calendar, left_on='order_date', right_on='order_date')
    calendar_loc = calendar.loc[(calendar['order_date'] <= (predict_cur_date + timedelta(span))) \
                                & (calendar['order_date'] >= predict_cur_date - timedelta(span + 1)), :]
    holidays = flatten_holiday()
    lunar_map = {}
    solar_map = {}
    holiday_map = {}
    calendar_loc = pd.merge(calendar_loc, holidays, left_on='order_date', right_on='order_date', how='left')
    calendar_loc['holiday'].fillna(0, inplace=True)
    for _, row in calendar_loc.iterrows():
        lunar_map[row['order_date']] = int(float(row['lunar_week']))
        solar_map[row['order_date']] = int(float(row['solar_week']))
        holiday_map[row['order_date']] = int(float(row['holiday']))

    sku_lunar_ratio_map = dict(df[['lunar_week', 'original_sale_num']] \
                               .groupby('lunar_week') \
                               .mean() \
                               .reset_index() \
                               .values \
                               .tolist())
    sku_solar_ratio_map = dict(df[['solar_week', 'original_sale_num']] \
                               .groupby('solar_week') \
                               .mean() \
                               .reset_index() \
                               .values \
                               .tolist())
    df = pd.merge(df_agg, holidays, left_on='order_date', right_on='order_date', how='left')
    df['holiday'].fillna(0, inplace=True)
    sku_holiday_ratio_map = dict(df[['holiday', 'original_sale_num']] \
                                 .groupby('holiday') \
                                 .mean() \
                                 .reset_index() \
                                 .values \
                                 .tolist())
    solar_ratio_map, \
    lunar_ratio_map, \
    brand_solar_ratio_map, \
    brand_lunar_ratio_map, \
    holiday_ratio_map, \
    brand_holiday_ratio_map = get_ratio_map(keys, ratio_info)

    debug_print('ratio_map', debug)
    debug_print('solar_ratio_map:{0}'.format(solar_ratio_map), debug)
    debug_print('lunar_ratio_map:{0}'.format(lunar_ratio_map), debug)
    debug_print('brand_solar_ratio_map:{0}'.format(brand_solar_ratio_map), debug)
    debug_print('brand_lunar_ratio_map:{0}'.format(brand_lunar_ratio_map), debug)
    debug_print('sku_solar_ratio_map:{0}'.format(sku_solar_ratio_map), debug)
    debug_print('sku_lunar_ratio_map:{0}'.format(sku_lunar_ratio_map), debug)
    debug_print('holiday_ratio_map:{0}'.format(holiday_ratio_map), debug)
    debug_print('brand_holiday_ratio_map:{0}'.format(brand_holiday_ratio_map), debug)
    debug_print('sku_holiday_ratio_map:{0}'.format(sku_holiday_ratio_map), debug)

    solar_ratio_map = smooth_dict(solar_ratio_map)
    lunar_ratio_map = smooth_dict(lunar_ratio_map)
    brand_solar_ratio_map = smooth_dict(brand_solar_ratio_map)
    brand_lunar_ratio_map = smooth_dict(brand_lunar_ratio_map)
    sku_solar_ratio_map = smooth_dict(sku_solar_ratio_map)
    sku_lunar_ratio_map = smooth_dict(sku_lunar_ratio_map)
    holiday_ratio_map = smooth_dict(holiday_ratio_map)
    brand_holiday_ratio_map = smooth_dict(brand_holiday_ratio_map)
    sku_holiday_ratio_map = smooth_dict(sku_holiday_ratio_map)

    ratio_span = 28
    adjusted_ratio_list = []
    for i in range(span):
        adjusted_ratio_list.append([])
    pivot_ratio_list = []
    lunar_on = False  # apply_lunar_seasonal(keys)
    lunar_std = np.std(list(lunar_map.values()))
    solar_std = np.std(list(solar_map.values()))
    if (lunar_std > solar_std):
        lunar_on = True
    solar_on = (not lunar_on)
    brand_lunar_on = ((lunar_on) & (len(brand_lunar_ratio_map.items()) >= 20))
    brand_solar_on = ((solar_on) & (len(brand_solar_ratio_map.items()) >= 50))
    sku_lunar_on = ((lunar_on) & (len(sku_lunar_ratio_map.items()) >= 20))
    sku_solar_on = ((solar_on) & (len(sku_solar_ratio_map.items()) >= 50))
    sku_holiday_on = (df_agg.shape[0] >= 360)
    for (calendar_map, ratio_map, is_on) in [(solar_map, solar_ratio_map, solar_on), \
                                             (lunar_map, lunar_ratio_map, lunar_on), \
                                             (solar_map, brand_solar_ratio_map, brand_solar_on), \
                                             (lunar_map, brand_lunar_ratio_map, brand_lunar_on), \
                                             (solar_map, sku_solar_ratio_map, sku_solar_on), \
                                             (lunar_map, sku_lunar_ratio_map, sku_lunar_on), \
                                             (holiday_map, holiday_ratio_map, True), \
                                             (holiday_map, brand_holiday_ratio_map, True), \
                                             (holiday_map, sku_holiday_ratio_map, sku_holiday_on)]:
        if (not is_on):
            continue
        init_ratio_list = []
        ratio_mean = harmonious_mean(ratio_map.values())
        for i in range(-ratio_span, 0, 1):
            dt = predict_cur_date + timedelta(i)
            pivot_week = calendar_map[dt]
            init_ratio = ratio_map.get(pivot_week, np.NaN)
            if np.isnan(init_ratio):
                init_ratio = ratio_mean
            init_ratio_list.append(init_ratio)
        pivot_ratio = robust_op(init_ratio_list)
        if pivot_ratio <= 0.0:
            continue
        pivot_ratio_list.append(pivot_ratio)
        for i in range(span):
            dt = predict_cur_date + timedelta(i)
            pivot_week = calendar_map[dt]
            ratio = ratio_map.get(pivot_week, np.NaN)
            if np.isnan(ratio):
                ratio = ratio_mean
            adjusted_ratio_list[i].append(ratio)
    pivot_ratio = robust_op(pivot_ratio_list)
    if pivot_ratio <= 0.0:
        return predict_results_day
    pivot_sale = robust_op(predict_results_day[:ratio_span])
    original_sale_mean = robust_op(
        df_agg.loc[df_agg['order_date'] >= (predict_cur_date - timedelta(ratio_span)), 'sale_num'])
    pivot_sale = robust_op([pivot_sale, original_sale_mean])
    is_seasonal = is_seasonal_category(keys, ratio_info)
    ratio_list = []
    for i in range(span):
        dt = predict_cur_date + timedelta(i)
        ratio = robust_op(adjusted_ratio_list[i])
        cur_ratio = ratio / pivot_ratio
        ratio_list.append(cur_ratio)
    ratio_list = remove_outliers(ratio_list)
    ratio_upper = (5.0 if is_seasonal else seasonal_upper_threshold(keys))
    ratio_lower = (0.1 if is_seasonal else seasonal_lower_threshold(keys))
    ratio_mean = robust_op(ratio_list)
    if ratio_mean > 0.0 and ratio_mean < ratio_lower:
        for i in range(len(ratio_list)):
            ratio_list[i] = min(5.0, max(0.1, ratio_list[i] * (ratio_lower / ratio_mean)))
    elif ratio_mean > ratio_upper:
        for i in range(len(ratio_list)):
            ratio_list[i] = min(5.0, max(0.1, ratio_list[i] * (ratio_upper / ratio_mean)))
    for i in range(span):
        predict_results_day[i] = predict_results_day[i] * ratio_list[i]
        dt = predict_cur_date + timedelta(i)
        debug_print('cur_ratio:{0},dt:{1}'.format(ratio_list[i], dt), debug)
    debug_print('predict after tunning', debug)
    debug_print(predict_results_day, debug)
    return predict_results_day


def in_stock_mean_sale(cur_pos, stock_flag, sales, back_span=14, forward=14):
    back_pos = max(0, cur_pos - back_span)
    forward_pos = min(len(stock_flag), cur_pos + forward)
    stock_frag = stock_flag[back_pos:cur_pos] + stock_flag[cur_pos + 1:forward_pos]
    sales_frag = sales[back_pos:cur_pos] + sales[cur_pos + 1:forward_pos]

    if robust_op(stock_frag, op=np.nansum) / len(stock_frag) <= 0.3:
        if robust_op(stock_flag, op=np.nansum) / len(stock_flag) < 0.5:
            return harmonious_mean(sales_frag)
        else:
            return np.matmul(sales, stock_flag) / robust_op(stock_flag, op=np.nansum)
    else:
        return np.matmul(sales_frag, stock_frag) / robust_op(stock_frag, op=np.nansum)


def in_stock_mean_sale_external(cur_pos, stock_flag, sales, back_span=14, forward_span=14):
    back_pos = max(0, cur_pos - back_span)
    forward_pos = min(len(stock_flag), cur_pos + forward_span)
    stock_frag = stock_flag[back_pos:cur_pos] + stock_flag[cur_pos + 1:forward_pos]
    sales_frag = sales[back_pos:cur_pos] + sales[cur_pos + 1:forward_pos]

    if robust_op(stock_frag, op=np.nansum) / len(stock_frag) <= 0.8:
        if robust_op(stock_flag, op=np.nansum) / len(stock_flag) < 0.9:

            if len(sales_frag) > 1:
                forward_index = range(back_pos - cur_pos, 0)
                backward_index = range(1, forward_pos - cur_pos)
                forward_index = [-xx for xx in forward_index]
                index_list = forward_index + backward_index
                weight_list = [1.0 / math.sqrt(xx) for xx in index_list]
                result1 = np.matmul(sales_frag, weight_list) / 2.0
                # return harmonious_mean(sales_frag)
            else:
                result1 = np.max(sales_frag)

            sales_before_max = np.nanmax(sales[:cur_pos] + [0])
            result2 = np.matmul(sales_frag, stock_frag) / robust_op(stock_frag, op=np.nansum)
            result3 = np.matmul(sales, stock_flag) / robust_op(stock_flag, op=np.nansum)
            result = np.nanmax([result1, result2, result3])
            result = np.nanmin([sales_before_max, result])
            if result // result3 >= 3:
                result = np.nanmean([result2, result3])
            if result // result2 >= 3:
                result = np.nanmax([result2, result3])
            return result

        else:
            return np.matmul(sales, stock_flag) / robust_op(stock_flag, op=np.nansum)
    else:
        return np.matmul(sales_frag, stock_frag) / robust_op(stock_frag, op=np.nansum)


def out_stock_smooth(keys, df, stock_col, sale_col, back_span=14, forward_span=14):
    df['stock_flag'] = np.where(((df[stock_col] > 0) | (df[sale_col] > 0)), 1, 0)
    stock_flag = df['stock_flag'].tolist()
    sales = df[sale_col].tolist()
    new_sales = [0.0] * len(sales)
    for i in range(len(sales)):
        if stock_flag[i] == 0:
            new_sales[i] = 0.0
            if keys[keys_columns.index('item_third_cate_cd')] not in ['13673', '12215']:
                new_sales[i] = in_stock_mean_sale(i, stock_flag, sales, back_span, forward_span)
            else:
                back_span = 90
                forward_span = 90
                new_sales[i] = in_stock_mean_sale_external(i, stock_flag, sales, back_span, forward_span)
        else:
            new_sales[i] = sales[i]
    return new_sales


def baseline_smooth_frag(ts, absolute_gap, relative_gap):
    if (len(ts)) == 0:
        return ts
    elif len(ts) <= 3:
        return [harmonious_mean([np.min(ts), np.median(ts)])] * len(ts)

    best_quantile = np.NaN
    cover_cnt = np.NaN
    for i in range(60, 101):
        quantile = np.percentile(ts, i, interpolation='nearest')
        cnt = 0
        for t in ts:
            if (-absolute_gap[0] <= t - quantile <= absolute_gap[1]) \
                    or (quantile * (1.0 - relative_gap[0]) <= t <= quantile * (1.0 + relative_gap[1])):
                cnt += 1
        if cover_cnt is np.NaN or cnt > cover_cnt:
            cover_cnt = cnt
            best_quantile = quantile
    for i in range(len(ts)):
        t = ts[i]
        if (t > best_quantile + absolute_gap[1]) \
                and (t > best_quantile * (1.0 + relative_gap[1])):
            ts[i] = max(best_quantile + absolute_gap[1], best_quantile * (1.0 + relative_gap[1]))
        # elif (t < best_quantile - absolute_gap[0]) \
        #        and (t < best_quantile * (1.0 - relative_gap[0])):
        #    ts[i] = min(best_quantile - absolute_gap[0], best_quantile * (1.0 - relative_gap[0]))
    return ts


def baseline_smooth_step(keys):
    step = 56
    try:
        if keys[keys_columns.index('item_third_cate_cd')] in ['736', '682']:
            # 682 - sanreqi 736 - zhilei
            step = 14
        elif keys[keys_columns.index('item_first_cate_cd')] in ['12218', '1315']:
            step = 28
        elif keys[keys_columns.index('item_first_cate_cd')] in ['652', '670', '12259']:
            step = 42
        elif keys[keys_columns.index('item_first_cate_cd')] in ['1320']:
            step = 112
    except:
        step = 56
    return step


def baseline_smooth(ts, step=28, absolute_gap=(2.0, 2.0), relative_gap=(0.5, 0.4)):
    if len(ts) < step * 2:
        return baseline_smooth_frag(ts, absolute_gap, relative_gap)
    frag_num = len(ts) // step
    new_ts = []
    for i in range(frag_num):
        back_index = i * step
        forward_index = (i + 1) * step
        if forward_index + step > len(ts):
            ts_frag = ts[back_index:]
        else:
            ts_frag = ts[back_index:forward_index]
        ts_frag = baseline_smooth_frag(ts_frag, absolute_gap, relative_gap)
        new_ts.extend(ts_frag)
    return new_ts


def apply_conservative_forecast(keys):
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_second_cate_cd')] in []) \
               | (keys[keys_columns.index('item_third_cate_cd')] in ['13576']))
    except:
        ans = False
    return ans


def big_promotion_smooth(keys, df, dt_key, stock_col, sale_col):
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_second_cate_cd')] in ['1523', '1525']) \
               | (keys[keys_columns.index('item_third_cate_cd')] in ['13576']))
    except:
        ans = False
    if ans:
        return df
    df['month'] = df[dt_key].map(lambda x: x.month)
    df['day'] = df[dt_key].map(lambda x: x.day)
    df['is_big_promotion'] = 0
    df.loc[((df['month'] == 6) & (1 <= df['day']) & (df['day'] <= 18)), 'is_big_promotion'] = 1
    df.loc[((df['month'] == 11) & (1 <= df['day']) & (df['day'] <= 11)), 'is_big_promotion'] = 1
    df_non_promo = df[df['is_big_promotion'] == 0]
    df_non_promo.reset_index(inplace=True)
    if df_non_promo.shape[0] >= 7:
        sale_mean = np.nanmean(baseline_smooth(df_non_promo[-28:][sale_col].tolist(), step=14))
        if sale_mean == 0:
            sale_mean = np.nanmean(baseline_smooth(df_non_promo[-28 * 2:][sale_col].tolist(), step=14))
    else:
        sale_mean = np.nanmean(baseline_smooth(df[sale_col].tolist(), step=14))
    df.loc[((df['month'] == 6) & (1 <= df['day']) & (df['day'] <= 18)), sale_col] = sale_mean
    df.loc[((df['month'] == 11) & (1 <= df['day']) & (df['day'] <= 11)), sale_col] = sale_mean
    return df


def apply_lunar_seasonal(keys):
    ans = False
    try:
        ans = ((keys[keys_columns.index('item_second_cate_cd')] in ['12259', '1320', '12218']) \
               | (keys[keys_columns.index('item_second_cate_cd')] in []) \
               | (keys[keys_columns.index('item_first_cate_cd')] in []))
    except:
        ans = False
    return ans


def result_ensemble_by_mapd(predict_list, accuracy_list):
    predict_min = min(predict_list)
    predict_max = max(predict_list)
    if predict_max / (predict_min + 1.0) >= 5.0:
        return harmonious_mean(predict_list)
    total = np.matmul(predict_list, accuracy_list)
    total_weight = np.sum(accuracy_list)
    if total_weight == 0.0:
        return harmonious_mean(predict_list)
    else:
        return total / total_weight


def sale_forecast_lr(keys, df_agg, df_promotion, predict_cur_date, \
                     debug=False, jd_price_adjusted=False, \
                     ratio_info=None, calendar=None):
    lr_period = 7
    if predict_cur_date <= datetime(predict_cur_date.year, 6, 18):
        start_dt_618 = datetime(predict_cur_date.year - 1, 6, 1)
        end_dt_618 = datetime(predict_cur_date.year - 1, 6, 18)
    else:
        start_dt_618 = datetime(predict_cur_date.year, 6, 1)
        end_dt_618 = datetime(predict_cur_date.year, 6, 18)
    start_dt_11 = datetime(predict_cur_date.year - 1, 11, 1)
    end_dt_11 = datetime(predict_cur_date.year - 1, 11, 14)
    pre_dt_618 = start_dt_618 - timedelta(recent_day_num)
    pre_dt_11 = start_dt_11 - timedelta(recent_day_num)
    dt_11 = datetime(predict_cur_date.year, 11, 11)
    dt_618 = datetime(predict_cur_date.year, 6, 18)
    df_big_promotion = df_agg.loc[((df_agg.order_date <= end_dt_618) & (df_agg.order_date >= start_dt_618)) \
                                  | ((df_agg.order_date <= end_dt_11) & (df_agg.order_date >= start_dt_11)), :]
    df_pre_big_promotion = df_agg.loc[((df_agg.order_date <= start_dt_618) & (df_agg.order_date >= pre_dt_618)) \
                                      | ((df_agg.order_date <= start_dt_11) & (
            df_agg.order_date >= pre_dt_11)), 'sale_num']
    if df_big_promotion.shape[0] < 5:
        sale_upper = np.percentile(np.unique(df_agg.sale_num), 90)
        sale_lower = np.percentile(np.unique(df_agg.sale_num), 60)
        df_big_promotion = df_agg.loc[(df_agg.sale_num <= sale_upper) & (df_agg.sale_num >= sale_lower), :]
    sale_mean_pre_big_promotion = np.NaN
    if df_big_promotion.shape[0] >= 5:
        sale_mean_big_promotion = df_big_promotion.sale_num.mean()
        if df_pre_big_promotion.shape[0] >= 10:
            sale_mean_pre_big_promotion = harmonious_mean(df_pre_big_promotion)
    else:
        sale_mean_big_promotion = df_agg.sale_num.mean()

    df_agg.loc[:, 'original_sale_num'] = df_agg['sale_num']
    df_agg = big_promotion_smooth(keys, df_agg, 'order_date', 'stock_qtty', 'sale_num')
    debug_print('original sales', debug)
    debug_print(df_agg.sale_num, debug)
    adj_mean = harmonious_mean(df_agg.sale_num)
    debug_print('adj_mean: {0}'.format(adj_mean), debug)
    df_agg['sale_num'].fillna(adj_mean, inplace=True)
    df_agg['original_sale_num'].fillna(adj_mean, inplace=True)

    in_stock_sales = df_agg.loc[((df_agg['stock_qtty'] > 0) | (df_agg['sale_num'] > 0)), :]['sale_num'].tolist()
    debug_print('in_stock_sales: {0}'.format(in_stock_sales), debug)
    smooth_step = baseline_smooth_step(keys)
    is_seasonal = is_seasonal_category(keys, ratio_info)
    if is_seasonal:
        smooth_step = 14
    debug_print('smooth step: %i' % smooth_step, debug)
    in_stock_sales = baseline_smooth(in_stock_sales, step=smooth_step)
    df_agg.loc[((df_agg['stock_qtty'] > 0) | (df_agg['sale_num'] > 0)), 'sale_num'] = in_stock_sales
    debug_print('in_stock_sales: {0}' \
                .format(df_agg.loc[((df_agg['stock_qtty'] > 0) | (df_agg['sale_num'] > 0)), :]['sale_num']), debug)
    df_agg.loc[:, 'sale_num'] = out_stock_smooth(keys, df_agg, 'stock_qtty', 'sale_num')
    df_agg['original_sale_num'].fillna(df_agg['sale_num'], inplace=True)
   
	#enter into the long tail model
    if keys[keys_columns.index('item_first_cate_cd')] in ['737','652'] and len(in_stock_sales) > 28 and robust_op(in_stock_sales) > 0.05 and robust_op(in_stock_sales) <= 0.3:

        #compute some eigenvalues on in_stock_sales
        long_tail_max = np.nanmax(in_stock_sales)
        long_tail_min = np.nanmin(in_stock_sales)
        long_tail_mean = np.nanmean(in_stock_sales)
        long_tail_std = np.nanstd(in_stock_sales)

        #compute the threshold for the first bin
        long_tail_interval = (long_tail_max - long_tail_min)/10.0
        long_tail_ratio_threshold = long_tail_min + long_tail_interval
        long_tail_abs_threshold = 2.0
        long_tail_percentile_threshold = np.percentile(in_stock_sales,10)
        long_tail_threshold = np.nanmin([long_tail_ratio_threshold,long_tail_abs_threshold,long_tail_percentile_threshold])
        
        #calculate the long tail ratio for identifying the long tail SKU
        long_tail_ratio = float(sum(xx<=long_tail_threshold for xx in in_stock_sales))/len(in_stock_sales)
        
        if long_tail_ratio >= 0.70:
            
            #compute the ratio of small sales in start four weeks
            in_stock_start_list = in_stock_sales[:28]
            #calculate some eigenvalues on the in_stock_start list
            long_tail_start_mean = np.nanmean(in_stock_start_list)
            long_tail_start_std = np.nanstd(in_stock_start_list)

            #compute the ratio of small sales within recent four weeks
            in_stock_recent_list = in_stock_sales[-28:]
            #calculate some eigenvalues on the in_stock_recent_sales
            long_tail_recent_mean = np.nanmean(in_stock_recent_list)
            long_tail_recent_std = np.nanstd(in_stock_recent_list)

            if long_tail_recent_mean < long_tail_start_mean:

                return [robust_op(in_stock_recent_list)] * predict_periods_num + [-1.0, 0.0, 'long_tail_model']

    elif robust_op(in_stock_sales) <= 0.05:
        debug_print('long tail forecast', debug)
        debug_print('in stock sales mean: {0}'.format(robust_op(in_stock_sales)), debug)
        return [robust_op(in_stock_sales)] * predict_periods_num + [-1.0, 0.0, 'long_tail_model']

    df_agg_recent = df_agg.loc[df_agg['order_date'] >= (predict_cur_date - timedelta(60)), :]
    # no valid history data
    if df_agg.shape[0] == 0:
        return [0.0] * predict_periods_num + [-1.0, 0.0]

    df_agg.loc[(df_agg['stock_qtty'] <= 0) \
               & (df_agg['original_sale_num'] <= 0), 'original_sale_num'] = np.NaN

    fb_forecast_original = fb_forecast = [np.NaN] * (predict_periods_num * period)
    fb_forecast, fb_mapd = prophet_forecast(keys, df_agg, \
                                            predict_cur_date, 'order_date', 'sale_num')
    day_sales = df_agg['sale_num'].tolist()
    day_sales = np.array(baseline_smooth(day_sales, step=smooth_step))
    debug_print('sale_num_len: %i' % (df_agg.shape[0]), debug)
    debug_print('day_sales_len: %i' % len(day_sales), debug)
    df_agg.loc[:, 'sale_num'] = day_sales

    # date_threshold = predict_cur_date - timedelta(recent_day_num + period)
    debug_print(day_sales, debug)
    day_sales_recent = df_agg['sale_num'][-(recent_day_num + period):]
    recent_sale_mean = harmonious_mean(day_sales_recent)
    debug_print('day_sales_recent: {0}'.format(day_sales_recent), debug)
    period_sales = []
    df_agg['sale_num'].fillna(recent_sale_mean, inplace=True)
    original_day_sales = np.array(df_agg['sale_num'].tolist())
    period_num = len(original_day_sales) // lr_period
    for i in range(period_num):
        period_sales.append(robust_op(original_day_sales[lr_period * i: lr_period * (i + 1)], op=np.nansum))
    period_sales = np.array(period_sales)
    debug_print('original_day_sales:{0}'.format(original_day_sales), debug)
    debug_print('period_sales:{0}'.format(period_sales), debug)
    quanreg_results = None
    if (len(day_sales) < records_num_threshold * lr_period) or (adj_mean <= 0.5):
        predict_results = []
        if len(day_sales_recent) >= recent_day_num:
            sales_to_be_sampled = day_sales_recent
        else:
            sales_to_be_sampled = (day_sales if len(day_sales) < recent_day_num else day_sales[-recent_day_num:])
        sampled_num = min(lr_period, len(sales_to_be_sampled))
        predict_results.extend(weighted_avg(sales_to_be_sampled, (predict_periods_num * period) / lr_period, lr_period))
        predict_results.append(-0.5)
        predict_results.append(0.0)
    else:
        # period_sales = remove_outliers(period_sales)
        total_sales_periods = robust_op([1 if s > 0.0 else 0 for s in period_sales], op=np.nansum)
        total_periods = len(period_sales)
        num_periods_year = 365 // lr_period
        if (total_periods >= num_periods_year + records_num_threshold):
            history_periods = 2
            latest_periods_index = [-i for i in range(1, history_periods + 1)]
            seasonal_periods_index = \
                [-i for i in range(num_periods_year, num_periods_year - history_periods, -1)]
            history_periods = num_periods_year
            seasonal = True
        else:
            seasonal_periods_index = []
            history_periods = 3
            latest_periods_index = [-i for i in range(1, history_periods + 1)]
            seasonal = False
        history_periods_index = latest_periods_index + seasonal_periods_index
        history_records = []
        targets = []
        for i in range(history_periods, total_periods):
            item = []
            for j in history_periods_index:
                item.append(period_sales[i + j])
            targets.append(period_sales[i])
            # item.append(1.0)
            history_records.append(item)
        # Sample the best LR from Ridge and Lasso with different reguralization parameters
        mapd = np.NaN
        r2 = np.NaN
        weights = np.NaN
        weight_max = np.NaN
        intercept = np.NaN
        regressor = np.NaN
        for estimator in [Ridge, Lasso]:
            for alpha in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
                for lr_method in [linear_regression, ransac_regression]:
                    for is_intercept in [False]:
                        try:
                            targets_hat, cur_r2, cur_weights, cur_intercept, cur_regressor \
                                = lr_method(estimator, history_records, targets, \
                                            alpha, intercept=is_intercept)
                            cur_mapd = compute_mapd(targets, targets_hat)
                            if mapd is np.NaN or cur_mapd < mapd:
                                mapd = cur_mapd
                                r2 = cur_r2
                                weights = cur_weights
                                intercept = cur_intercept
                                regressor = cur_regressor
                                weight_max = max([abs(w) for w in weights])
                        except:
                            continue
            debug_print('weights: {0}, intercept: {1}, r2: {2}, mapd: {3}' \
                        .format(weights, intercept, r2, mapd), debug)
        predict_results = []
        if quantile_regression_on:
            quanreg_results = quantile_regression(history_records, \
                                                  targets, \
                                                  (predict_periods_num * period) // lr_period, \
                                                  period_sales, \
                                                  seasonal, \
                                                  latest_periods_index, \
                                                  seasonal_periods_index, \
                                                  qs=np.arange(.4, .6, .01))
        # customized regularization
        if r2 >= 0.5 and (0.9 <= abs(sum(weights)) <= 1.1) and weight_max < 1.5:
            features = period_sales[history_periods_index]
            # features = np.append(features,1.0)
            features = np.array(features)
            for i in range((predict_periods_num * period) // lr_period):
                result = max(0.0, regressor.predict([features])[0])
                debug_print('{0}-th predict: features: {1}, result:{2}'.format(i, features, result), debug)
                if i > 0:
                    result = harmonious_mean([result, predict_results[0]])
                predict_results.append(result)
                if seasonal:
                    his_predict_sales = np.append(period_sales, predict_results)
                    features = np.append(his_predict_sales[latest_periods_index], \
                                         his_predict_sales[seasonal_periods_index])
                else:
                    features = np.roll(features, 1)
                    features[0] = result
            predict_results.append(r2)
            predict_results.append(mapd)
        else:
            if len(day_sales_recent) >= recent_day_num:
                sales_to_be_sampled = day_sales_recent
            else:
                sales_to_be_sampled = (day_sales if len(day_sales) < recent_day_num else day_sales[-recent_day_num:])
            sampled_num = min(period, len(sales_to_be_sampled))
            predict_results.extend(
                weighted_avg(sales_to_be_sampled, (predict_periods_num * period) / lr_period, lr_period))
            mapd = compute_mapd(targets[-(predict_periods_num * period) / lr_period:], \
                                np.array(predict_results[-(predict_periods_num * period) / lr_period:]))
            predict_results.extend([0.0, mapd])

    predict_results = tune_results_by_quanreg(keys, predict_results, quanreg_results)

    predict_results_day = np.array([0.0] * (period * predict_periods_num))
    for i in range(0, period * predict_periods_num):
        predict_results_day[i] = float(predict_results[i // lr_period] / lr_period)

    tsfresh_predict_results_day, tsfresh_mapd = tsfresh_forecast(keys, \
                                                                 df_agg, \
                                                                 df_promotion, \
                                                                 predict_results_day, \
                                                                 predict_cur_date, \
                                                                 ratio_info, \
                                                                 calendar, \
                                                                 debug)
    # set no limit on categories. modified by wusai1 on 2019/01/29
    if df_agg['order_date'].min() <= (predict_cur_date - pd.DateOffset(months=16)):
        # print(df_agg[df_agg.order_date == '2018-02-06'][['order_date', 'sale_num', 'original_sale_num']])
        df_agg_new = chinese_new_year_correspond(df_agg, spring_festivals, predict_cur_date, debug)
        # print(df_agg_new[df_agg.order_date == '2018-02-06'][['order_date', 'sale_num', 'original_sale_num']])
        seasonal_predict_results_day, seasonal_mapd = lr_seasonal_forecast(keys, \
                                                                       df_agg_new, \
                                                                       df_promotion, \
                                                                       predict_results_day, \
                                                                       predict_cur_date, \
                                                                       ratio_info, \
                                                                       calendar, \
                                                                       debug)

        del df_agg_new
    else:
        seasonal_predict_results_day, seasonal_mapd = lr_seasonal_forecast(keys, \
                                                                           df_agg, \
                                                                           df_promotion, \
                                                                           predict_results_day, \
                                                                           predict_cur_date, \
                                                                           ratio_info, \
                                                                           calendar, \
                                                                           debug)

    debug_print('seasonal_forecast: {0}'.format(seasonal_predict_results_day), debug)
    model_name = 'lr_quanreg'
    if not (np.isnan(tsfresh_mapd)):
        model_name += '_lgb_fresh'
    if not (np.isnan(fb_mapd)):
        model_name += '_prophet'
    for i in range(predict_periods_num * period):
        accuracy_list = [0.5]
        lr_forecast_result = predict_results_day[i]
        forecast_list = [lr_forecast_result]
        fb_forecast_result = fb_forecast[i]
        tsfresh_forecast_result = tsfresh_predict_results_day[i]
        if not (np.isnan(fb_forecast_result) or fb_forecast_result <= 0.0):
            accuracy_list.append(0.0 if fb_mapd >= 1.0 else 1 - fb_mapd)
            forecast_list.append(fb_forecast_result)
        if not (np.isnan(tsfresh_forecast_result) or tsfresh_forecast_result <= 0.0):
            accuracy_list.append(0.0 if tsfresh_mapd >= 1.0 else 1 - tsfresh_mapd)
            forecast_list.append(tsfresh_forecast_result)
        debug_print('i:{0},acc: {1}, fr:{2}'.format(i, accuracy_list, forecast_list), debug)
        predict_results_day[i] = result_ensemble_by_mapd(forecast_list, accuracy_list)

    predict_results_day = tune_results_by_ratio(keys, \
                                                df_agg, \
                                                predict_results_day, \
                                                predict_cur_date, \
                                                ratio_info, \
                                                calendar, \
                                                debug)

    predict_results_day = tune_results_by_promotion(keys, \
                                                    df_agg, \
                                                    df_promotion, \
                                                    predict_results_day, \
                                                    predict_cur_date, \
                                                    debug)

    if not (seasonal_mapd is np.NaN):
        model_name += '_seasonal'
        debug_print('seasonal_predict_results: {0}'.format(seasonal_predict_results_day), debug)
    if is_seasonal:
        model_name += '_catesea'

    if not (seasonal_mapd is np.NaN):
        if apply_seasonal_overwrite(keys):
            predict_results_day = seasonal_predict_results_day
        else:
            for i in range(predict_periods_num * period):
                if seasonal_predict_results_day[i] >= 0.0:
                    predict_results_day[i] = np.mean([predict_results_day[i], seasonal_predict_results_day[i]])

    # mainly for fresh food
    if apply_conservative_forecast(keys) \
            and len(period_sales) > 0 \
            and (period_sales[-1] < robust_op(predict_results_day[:lr_period], op=np.nansum)):
        result = harmonious_mean([period_sales[-1], robust_op(predict_results_day[:lr_period], op=np.nansum)])
        for i in range(lr_period):
            predict_results_day[i] = result / lr_period
    debug_print('recent_sale_mean: {0}'.format(recent_sale_mean), debug)
    if np.isnan(sale_mean_pre_big_promotion):
        sale_mean_pre_big_promotion = recent_sale_mean
    sale_mean_pre_big_promotion = max(1.0, sale_mean_pre_big_promotion)
    inc_ratio = max(1.5, min(3.0, sale_mean_big_promotion / sale_mean_pre_big_promotion))
    debug_print('sale_mean_big_promotion: {0}, sale_mean_pre_big_promotion: {1}, inc_ratio: {2}' \
                .format(sale_mean_big_promotion, sale_mean_pre_big_promotion, inc_ratio), debug)
    if apply_big_promotion(keys):
        if (dt_618 - predict_cur_date).days < period * predict_periods_num \
                or (dt_11 - predict_cur_date).days < period * predict_periods_num:
            debug_print('big promotion tuning: {0}'.format(sale_mean_big_promotion), debug)
            for i in range(0, period * predict_periods_num):
                predict_dt = predict_cur_date + timedelta(i)
                if (predict_dt <= dt_618 and predict_dt >= dt_618 - timedelta(2)) \
                        or (predict_dt <= dt_11 and predict_dt >= dt_11 - timedelta(2)):
                    predict_results_day[i] = inc_ratio * predict_results_day[i]

    predict_results = [0.0] * predict_periods_num
    for i in range(predict_periods_num):
        predict_results[i] = robust_op(predict_results_day[i * period:(i + 1) * period], op=np.nansum)

    max_period_sale = df_agg['sale_num'].max() * period  # np.max(period_sales)
    for i in range(len(predict_results)):
        if predict_results[i] > max_period_sale:
            predict_results[i] = harmonious_mean([predict_results[i], max_period_sale])
        if (predict_results[i] is None) or np.isnan(predict_results[i]):
            predict_results[i] = 0.0
    return predict_results + [0.0, 0.0, model_name]


def extract_ratio_info(record):
    ratio_info = None
    if len(record) >= 3 and not (record[2] is None):
        ratio_info = pd.DataFrame(data=record[2], columns=['key_name', 'week', 'ratio'])
    return ratio_info


def extract_sales_promotion_info(record):
    df_agg = None
    df_promotion = None
    for items in record[1]:
        start_date = datetime.strptime(items[timeseries_filtered_index('start_date')], '%Y-%m-%d')
        num_days = int(items[timeseries_filtered_index('len')])
        data_key = items[timeseries_filtered_index('data_key')]
        series = [try_parse(l) for l in
                  items[timeseries_filtered_index('y')].replace('[', '').replace(']', '').split(',')]
        end_date = start_date + timedelta(num_days - 1)
        date_series = pd.date_range(start_date, end_date)
        df_tmp = pd.DataFrame([[x, y] for x, y in zip(date_series, series)], \
                              columns=['order_date', data_key_columns_map[data_key]])
        if data_key in data_key_columns_map_promotion.keys():
            if df_promotion is None:
                df_promotion = df_tmp
            else:
                df_promotion = pd.merge(df_promotion, df_tmp, left_on='order_date', right_on='order_date', how='inner')
        else:
            if df_agg is None:
                df_agg = df_tmp
            else:
                df_agg = pd.merge(df_agg, df_tmp, left_on='order_date', right_on='order_date', how='outer')
    #####按可售状态进行填充
    if ('vendibility' in df_agg.columns):
        df_agg = df_agg.sort_values('order_date')
        if (len(df_agg) == 0) | (len(df_agg[df_agg.vendibility > 0]) == 0):
            return None, None, None
        vend_date = min(df_agg[df_agg.vendibility > 0].order_date)
        df_agg = df_agg[df_agg.order_date >= vend_date]
        # sale_date = min(df_agg[df_agg.sale_num.notnull()].order_date)
        # df_agg.loc[(df_agg.order_date<sale_date)&(df_agg.sale_num.isnull()),'sale_num'] = 0.0
        df_agg['sale_num'].fillna(0.0, inplace=True)
    else:
        df_agg = df_agg[df_agg.sale_num.notnull()]

    keys = record[0].split('#')
    if ('vendibility' in df_agg.columns) and apply_vendibility_smooth(keys):
        df_agg['vendibility'].fillna(method='ffill', inplace=True)
        df_agg.loc[:, 'stock_qtty'] = df_agg['vendibility']
        # df_agg.loc[df_agg['vendibility'] == 0,'stock_qtty'] = 0
    df_agg.avg_jd_unit_price.fillna(method='pad', inplace=True)
    df_agg.avg_jd_unit_price.fillna(method='ffill', inplace=True)
    jd_price_mean = harmonious_mean(df_agg['avg_jd_unit_price'])
    df_agg.avg_jd_unit_price.fillna(jd_price_mean, inplace=True)
    df_agg.avg_sale_unit_price.fillna(df_agg.avg_jd_unit_price, inplace=True)
    df_agg.stock_qtty.fillna(method='ffill', inplace=True)
    return keys, df_agg, df_promotion


def features_generator(record, predict_cur_date, \
                       debug=False, calendar=None):
    keys, df_agg, df_promotion = extract_sales_promotion_info(record)
    if df_agg.loc[(df_agg.stock_qtty > 0) | (df_agg.sale_num > 0.0), :].shape[0] == 0:
        return []
    calendar_info = (None if calendar is None else calendar.value)
    ratio = extract_ratio_info(record)
    ts_list = tsfresh_features_ts(keys, df_agg, df_promotion, \
                                  predict_cur_date, \
                                  ratio_info=ratio, \
                                  calendar=calendar_info, \
                                  debug=debug)
    return ts_list


# Return results contain the following info in sequence:
# sku_id,dc_id,predict results in @predict_periods_num periods, r2, mapd
def sale_forecast_adjusted(record, predict_cur_date, \
                           fr_period=7, fr_predict_periods_num=13, \
                           debug=False, jd_price_adjusted=False,
                           calendar=None, sc=None, \
                           weather=None, reset_weather_flag=True):
    if reset_weather_flag:
        global weather_as_features
        weather_as_features = False
    global period
    global predict_periods_num
    period = fr_period
    predict_periods_num = fr_predict_periods_num
    keys, df_agg, df_promotion = extract_sales_promotion_info(record)
    if df_agg is None:
        return record[0].split('#')[:2] + [0.0] * predict_periods_num + [-1.0, 0.0, 'out_stock_default_0']
    out_of_stock_threshold = get_out_of_stock_threshold(keys)
    out_of_stock_dt = predict_cur_date - timedelta(out_of_stock_threshold)
    if df_agg.loc[(df_agg.order_date >= out_of_stock_dt) \
                  & ((df_agg.stock_qtty > 0) | (df_agg.sale_num > 0)), :].shape[0] == 0:
        debug_print(df_agg.loc[(df_agg.order_date >= out_of_stock_dt), :], debug)
        debug_print('out of stock for more than {0} days'.format(out_of_stock_threshold), debug)
        return record[0].split('#')[:2] + [0.0] * predict_periods_num + [-1.0, 0.0, 'out_stock_default_0']

    earlist_date = predict_cur_date - timedelta(52 * 2 * 7)
    # earlist_date = min(earlist_date,df_agg.loc[((df_agg.stock_qtty > 0) | (df_agg.sale_num > 0)),'order_date'].min())
    df_agg = df_agg.loc[(df_agg.order_date >= earlist_date) & (df_agg.order_date < predict_cur_date), :]
    if df_agg.loc[(df_agg.stock_qtty > 0) | (df_agg.sale_num > 0.0), :].shape[0] == 0:
        debug_print('no valid records', debug)
        return record[0].split('#')[:2] + [0.0] * predict_periods_num + [-1.0, 0.0, 'no_valid_record_default_0']
    calendar_full = (None if calendar is None else calendar.value)
    dc_id = int(keys[keys_columns.index('dc_id')])
    if weather_as_features and (calendar_full is not None):
        dc_id_list = calendar_full[~(calendar_full.dc_id.isnull())].dc_id.drop_duplicates().tolist()
        dc_id_list = [int(d) for d in dc_id_list]
        if dc_id in dc_id_list:
            calendar_info = calendar_full[calendar_full['dc_id'] == dc_id]
        else:
            calendar_info = calendar_full[calendar_full['dc_id'] == 0.0]
        calendar_info.drop('dc_id', inplace=True, axis=1)
        print('calendar_info:', calendar_info.head(10))
    else:
        calendar_info = calendar_full
    ratio = extract_ratio_info(record)
    if debug:
        for (name, df) in [('sales', df_agg), \
                           ('promotion', df_promotion), \
                           ('ratio', ratio), \
                           ('calendar', calendar_info)]:
            if not (df is None):
                sc.parallelize([df.columns.tolist()] + df.values.tolist()) \
                    .map(lambda line: ','.join([str(l) for l in line])) \
                    .saveAsTextFile('/tmp/sl/lr/{0}/{1}'.format(record[0], name))
    predict_results = forecast_result_ensemble(keys, df_agg, df_promotion, \
                                               predict_cur_date, debug, \
                                               jd_price_adjusted, \
                                               ratio_info=ratio, \
                                               calendar=calendar_info)
    return record[0].split('#')[:2] + predict_results


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def prophet_forecast(keys, df, predict_cur_date, dt_key, val_key):
    results = [np.NaN] * (predict_periods_num * period)
    mapd = np.NaN
    if apply_fbprophet(keys) and df.shape[0] >= 180 and df[val_key].mean() >= 5.0:
        try:
            with suppress_stdout_stderr():
                m = Prophet(holidays=get_holidays(keys), yearly_seasonality=True)
                df_loc = df[[dt_key, val_key]]
                df_loc.rename(columns={dt_key: 'ds', val_key: 'y'}, inplace=True)
                m.fit(df_loc)
                future = m.make_future_dataframe(periods=predict_periods_num * period)
                forecast = m.predict(future)
                results = forecast.loc[forecast['ds'] >= predict_cur_date, 'yhat'].tolist()
                train_y = df_loc.loc[df_loc['ds'] < predict_cur_date, 'y'].tolist()
                train_y_hat = forecast.loc[forecast['ds'] < predict_cur_date, 'yhat'].tolist()
                mapd = compute_mapd(np.array(train_y), np.array(train_y_hat))
        except:
            print('prophet error')
    return np.array(results), mapd


def extract_features_from_series(df_sku, dt_key, val_key, sample_sales, lag=14, rolling_direction=1):
    forecast_span = predict_periods_num * period
    X_raw = df_sku[val_key].tolist()
    X_pred = np.random.choice(sample_sales, forecast_span)
    X_raw.extend(X_pred)
    start_date = df_sku[dt_key].min()
    X = pd.Series(data=X_raw, index=pd.date_range(start_date, periods=len(X_raw)))
    df_shift, y = make_forecasting_frame(X, kind=val_key, max_timeshift=lag, rolling_direction=rolling_direction)
    with suppress_stdout_stderr():
        X_fea = extract_features(df_shift, column_id="id", column_sort="time", \
                                 column_value="value", \
                                 impute_function=impute, show_warnings=False, \
                                 default_fc_parameters=MinimalFCParameters())
    #                         kind_to_fc_parameters=fc_parameters)
    for i in range(1, lag + 1, 1):
        X_fea["feature_last_%i_value" % i] = y.shift(i)
    X.index.name = 'id'
    X.name = 'target'
    df = X_fea.join(X)
    df = df.reset_index()
    df.rename(columns={'id': dt_key}, inplace=True)
    del X
    del X_fea
    return df


def get_ratio_map(keys, ratio_info):
    solar_ratio_map = {}
    lunar_ratio_map = {}
    brand_solar_ratio_map = {}
    brand_lunar_ratio_map = {}
    holiday_ratio_map = {}
    brand_holiday_ratio_map = {}
    dc_str = str(int(float(keys[keys_columns.index('dc_id')])))
    for _, row in ratio_info.iterrows():
        if row['key_name'] in ['solar_' + dc_str, 'solar_' + dc_str + '.0']:
            solar_ratio_map[int(float(row['week']))] = float(row['ratio'])
        elif row['key_name'] in ['lunar_' + dc_str, 'lunar_' + dc_str + '.0']:
            lunar_ratio_map[int(float(row['week']))] = float(row['ratio'])
        elif row['key_name'] in ['brand_solar_' + dc_str, 'brand_solar_' + dc_str + '.0']:
            brand_solar_ratio_map[int(float(row['week']))] = float(row['ratio'])
        elif row['key_name'] in ['brand_lunar_' + dc_str, 'brand_lunar_' + dc_str + '.0']:
            brand_lunar_ratio_map[int(float(row['week']))] = float(row['ratio'])
        elif row['key_name'] in ['holiday_' + dc_str, 'holiday_' + dc_str + '.0']:
            holiday_ratio_map[int(float(row['week']))] = float(row['ratio'])
        elif row['key_name'] in ['brand_holiday_' + dc_str, 'brand_holiday_' + dc_str + '.0']:
            brand_holiday_ratio_map[int(float(row['week']))] = float(row['ratio'])
    return solar_ratio_map, lunar_ratio_map, brand_solar_ratio_map, brand_lunar_ratio_map, holiday_ratio_map, brand_holiday_ratio_map


def error_weighted_mapd_lgb(preds, dtrain):
    labels = dtrain.get_label()
    return 'wmapd', compute_mapd(labels, preds), False


def train_model_lgb(train_data, valid_data, \
                    features, target, \
                    error_func=error_weighted_mapd_lgb):
    dataFeaturesTrain = train_data.loc[:, features]
    dataLabelsTrain = train_data.loc[:, target]
    dataFeaturesValid = valid_data.loc[:, features]
    dataLabelsValid = valid_data.loc[:, target]

    dTrain = lgb.Dataset(dataFeaturesTrain, dataLabelsTrain)
    dValid = lgb.Dataset(dataFeaturesValid, label=dataLabelsValid)

    params = {'task': 'train', \
              'boosting_type': 'gbdt', \
              'objective': 'regression', \
              'metric': {'rmse'}, \
              'num_leaves': 64, \
              'min_child_samples': 2, \
              'learning_rate': 0.1, \
              'feature_fraction': 0.7, \
              'bagging_fraction': 0.7, \
              'bagging_freq': 3, \
              'num_threads': 1, \
              'verbose': 1}  # 2018.12.3 add num_threads by ludq

    num_round = 30
    model = lgb.train(params, dTrain, num_boost_round=num_round, \
                      valid_sets=[], \
                      feval=error_func)
    preds_train = model.predict(dataFeaturesTrain, num_iteration=model.best_iteration)
    mapd = compute_mapd(dataLabelsTrain, preds_train)
    preds = model.predict(dataFeaturesValid, num_iteration=model.best_iteration)
    preds = np.array([max(0.0, p) for p in preds])
    return preds, mapd


def tsfresh_features_ts(keys, df_sku, df_promotion, predict_results_day, \
                        predict_cur_date, ratio_info, calendar, debug):
    df = tsfresh_features(keys, df_sku, df_promotion, predict_results_day, \
                          predict_cur_date, ratio_info, calendar, debug)
    if df is None:
        return []
    categorical_features = []
    categorical_features.append('holiday')
    categorical_features.append('solar_week')
    categorical_features.append('lunar_week')
    categorical_features.append('weekday')
    categorical_features.append('month')
    categorical_features.append('year')
    lstm_features = []
    continuous_features = df.columns.tolist()
    continuous_features.remove(dt_key)
    continuous_features.remove('target')
    for col in df.columns:
        if col.startswith('feature_last_'):  # or col.endswith('_ratio'):
            lstm_features.append(col)
    for col in lstm_features + categorical_features:
        continuous_features.remove(col)
    columns = ['target'] + continuous_features + lstm_features + categorical_features
    for col in columns:
        df[col].fillna(np.nanmedian(df[col]), inplace=True)
    df.fillna(0.0, inplace=True)
    ts_list = []
    for col in columns:
        tag = 'target'
        if col in continuous_features:
            tag = 'continuous'
        elif col in lstm_features:
            tag = 'lstm'
        elif col in categorical_features:
            tag = 'categorical'
        ts = '#'.join(keys + [col, predict_cur_date.strftime('%Y-%m-%d'), tag])
        ts += '#'
        ts += ','.join([str(l) for l in df[col].tolist()])
        ts_list.append(ts)
    return ts_list


def tsfresh_features(keys, df_sku, df_promotion, predict_results_day, \
                     predict_cur_date, ratio_info, calendar, debug):
    dt_key = 'order_date'
    val_key = get_target_sale_column(keys)
    id_key = 'sku_id'
    lag = 14
    cut_dt = datetime(2016, 1, 1)
    df_sku = df_sku.loc[((df_sku[dt_key] < predict_cur_date) \
                         & (df_sku[dt_key] >= cut_dt)), :]
    holidays = flatten_holiday()
    holidays.loc[:, dt_key] = pd.to_datetime(holidays[dt_key])
    sample_sales = df_sku.loc[~(df_sku[dt_key].isin(holidays[dt_key])), :][val_key].tolist()
    sample_sales = sample_sales[-28:]
    try:
        df = extract_features_from_series(df_sku, dt_key, val_key, sample_sales, lag=lag, rolling_direction=1)
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]
        df.loc[:, dt_key] = pd.to_datetime(df[dt_key])
        if not (calendar is None):
            calendar.loc[:, dt_key] = pd.to_datetime(calendar[dt_key])
            df = pd.merge(df, calendar, left_on=dt_key, right_on=dt_key)
        df = pd.merge(df, holidays, left_on=dt_key, right_on=dt_key, how='left')
        df['holiday'].fillna(0, inplace=True)
        if not (ratio_info is None):
            solar_ratio_map, \
            lunar_ratio_map, \
            brand_solar_ratio_map, \
            brand_lunar_ratio_map, \
            holiday_ratio_map, \
            brand_holiday_ratio_map = get_ratio_map(keys, ratio_info)
            solar_ratio_mean = robust_op(list(solar_ratio_map.values()))
            lunar_ratio_mean = robust_op(list(lunar_ratio_map.values()))
            brand_solar_ratio_mean = robust_op(list(brand_solar_ratio_map.values()))
            brand_lunar_ratio_mean = robust_op(list(brand_lunar_ratio_map.values()))
            holiday_ratio_mean = robust_op(list(holiday_ratio_map.values()))
            brand_holiday_ratio_mean = robust_op(list(brand_holiday_ratio_map.values()))
            if ('solar_week' in df.columns) and ('lunar_week' in df.columns):
                df['solar_ratio'] = df['solar_week'].map(lambda x: solar_ratio_map.get(x, solar_ratio_mean))
                df['lunar_ratio'] = df['lunar_week'].map(lambda x: lunar_ratio_map.get(x, lunar_ratio_mean))
                df['brand_solar_ratio'] = df['solar_week'].map(
                    lambda x: brand_solar_ratio_map.get(x, brand_solar_ratio_mean))
                df['brand_lunar_ratio'] = df['lunar_week'].map(
                    lambda x: brand_lunar_ratio_map.get(x, brand_lunar_ratio_mean))
            df['holiday_ratio'] = df['holiday'].map(lambda x: holiday_ratio_map.get(x, holiday_ratio_mean))
            df['brand_holiday_ratio'] = df['holiday'].map(
                lambda x: brand_holiday_ratio_map.get(x, brand_holiday_ratio_mean))
        df['weekday'] = df['order_date'].map(lambda x: x.weekday())
        df['month'] = df['order_date'].map(lambda x: x.month)
        df['year'] = df['order_date'].map(lambda x: x.year)
        df = merge_promotion_info(df, df_promotion, dt_key)
        df.replace([np.inf, -np.inf], np.NaN, inplace=True)
        cut_dt = predict_cur_date
        features = df.columns.tolist()
        features.remove('target')
        features.remove(dt_key)
        for col in features + ['target']:
            df[col].fillna(np.nanmedian(df[col]), inplace=True)
        df.fillna(0.0, inplace=True)
    except:
        df = None
        debug_print('tsfresh_features_error: {0}'.format(keys), debug)
    return df


def get_target_sale_column(keys):
    val_key = 'sale_num'
    try:
        if ((keys[keys_columns.index('item_first_cate_cd')] in []) \
                | (keys[keys_columns.index('item_second_cate_cd')] in []) \
                | (keys[keys_columns.index('item_third_cate_cd')] in ['1595', '736'])):
            val_key = 'original_sale_num'
    except:
        pass
    return val_key


def merge_promotion_info(df, df_promotion, dt_key):
    if df_promotion is not None:
        df_promotion[dt_key] = pd.to_datetime(df_promotion[dt_key])
        df = pd.merge(df, df_promotion, left_on=dt_key, right_on=dt_key, how='left')
        df['synthetic_discount'].fillna(0.0, inplace=True)
        df['synthetic_discount_round'] = df['synthetic_discount'].map(lambda x: round(x * 10))
        df['discount_mean'] = df[['synthetic_discount_round', 'target']] \
            .groupby('synthetic_discount_round') \
            .transform(np.nanmean)
        df['discount_ratio'] = df['discount_mean'] / (1.0 + np.nanmean(df['target']))
    return df


def lr_seasonal_features(keys, df_sku, val_key, df_promotion, predict_results_day, \
                         predict_cur_date, ratio_info, calendar, debug):
    lag = 2
    forecast_span = predict_periods_num * period
    cut_dt = datetime(2015, 1, 1)
    dt_key = 'order_date'
    df_sku = df_sku.loc[((df_sku[dt_key] < predict_cur_date) \
                         & (df_sku[dt_key] >= cut_dt)), :]
    X_raw = df_sku[val_key].tolist()
    X_pred = np.random.choice(X_raw[-14:], forecast_span)
    X_raw.extend(X_pred)
    start_date = df_sku[dt_key].min()
    data = []
    for i in range(lag, len(X_raw)):
        row = [X_raw[i - j] for j in range(lag + 1)]
        if i - 365 >= 0:
            row.append(X_raw[i - 365])
        else:
            row.append(np.NaN)
        row.append(start_date + timedelta(i))
        data.append(row)
    df = pd.DataFrame(data, columns=['target'] \
                                    + ['feature_last_%i_value' % i for i in range(1, lag + 1)] \
                                    + ['feature_last_year', dt_key])
    holidays = flatten_holiday()
    df.loc[:, dt_key] = pd.to_datetime(df[dt_key])
    holidays.loc[:, dt_key] = pd.to_datetime(holidays[dt_key])
    new_sku = (df.shape[0] < 120)
    if not (calendar is None):
        calendar.loc[:, dt_key] = pd.to_datetime(calendar[dt_key])
        if new_sku:
            calendar = calendar[((calendar[dt_key] < predict_cur_date + timedelta(forecast_span)) \
                                 & (calendar[dt_key] >= predict_cur_date - timedelta(365)))]
            df = pd.merge(calendar, df, left_on=dt_key, right_on=dt_key, how='left')
        else:
            df = pd.merge(df, calendar, left_on=dt_key, right_on=dt_key)
    df = pd.merge(df, holidays, left_on=dt_key, right_on=dt_key, how='left')
    df['holiday'].fillna(0, inplace=True)
    if not (ratio_info is None):
        solar_ratio_map, \
        lunar_ratio_map, \
        brand_solar_ratio_map, \
        brand_lunar_ratio_map, \
        holiday_ratio_map, \
        brand_holiday_ratio_map = get_ratio_map(keys, ratio_info)
        solar_ratio_mean = robust_op(list(solar_ratio_map.values()))
        lunar_ratio_mean = robust_op(list(lunar_ratio_map.values()))
        brand_solar_ratio_mean = robust_op(list(brand_solar_ratio_map.values()))
        brand_lunar_ratio_mean = robust_op(list(brand_lunar_ratio_map.values()))
        holiday_ratio_mean = robust_op(list(holiday_ratio_map.values()))
        brand_holiday_ratio_mean = robust_op(list(brand_holiday_ratio_map.values()))
        if ('solar_week' in df.columns) and ('lunar_week' in df.columns):
            df['solar_ratio'] = df['solar_week'].map(lambda x: solar_ratio_map.get(x, solar_ratio_mean))
            df['lunar_ratio'] = df['lunar_week'].map(lambda x: lunar_ratio_map.get(x, lunar_ratio_mean))
            df['brand_solar_ratio'] = df['solar_week'].map(
                lambda x: brand_solar_ratio_map.get(x, brand_solar_ratio_mean))
            df['brand_lunar_ratio'] = df['lunar_week'].map(
                lambda x: brand_lunar_ratio_map.get(x, brand_lunar_ratio_mean))
        df['holiday_ratio'] = df['holiday'].map(lambda x: holiday_ratio_map.get(x, holiday_ratio_mean))
        df['brand_holiday_ratio'] = df['holiday'].map(
            lambda x: brand_holiday_ratio_map.get(x, brand_holiday_ratio_mean))
        if new_sku:
            ratio_sum = np.nansum(df[~(df['target'].isnull())]['solar_ratio'])
            ratio_mean = np.nansum(df[~(df['target'].isnull())]['target']) / (1.0 if ratio_sum <= 0.0 else ratio_sum)
            df['target_filled'] = ratio_mean * df['solar_ratio']
            df['target'].fillna(df['target_filled'], inplace=True)
            df.drop('target_filled', axis=1, inplace=True)

    df['weekday'] = df[dt_key].map(lambda x: x.weekday())
    df['weekday_ratio'] = df[['weekday', 'target']].groupby('weekday') \
        .transform(lambda x: np.nanmean(x))
    df = merge_promotion_info(df, df_promotion, dt_key)
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    features = df.columns.tolist()
    features.remove('target')
    features.remove(dt_key)
    for col in features + ['target']:
        df[col].fillna(np.nanmedian(df[col]), inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def lr_seasonal_forecast(keys, df_sku, df_promotion, predict_results_day, \
                         predict_cur_date, ratio_info, calendar, debug):
    solar_results, solar_mapd \
        = lr_seasonal_forecast_base(keys, df_sku, 'sale_num', df_promotion, predict_results_day, \
                                    predict_cur_date, ratio_info, calendar, debug, is_solar=True)
    lunar_results, lunar_mapd \
        = lr_seasonal_forecast_base(keys, df_sku, 'sale_num', df_promotion, predict_results_day, \
                                    predict_cur_date, ratio_info, calendar, debug, is_solar=False)
    solar_results_org, solar_mapd_org \
        = lr_seasonal_forecast_base(keys, df_sku, 'original_sale_num', df_promotion, predict_results_day, \
                                    predict_cur_date, ratio_info, calendar, debug, is_solar=True)
    lunar_results_org, lunar_mapd_org \
        = lr_seasonal_forecast_base(keys, df_sku, 'original_sale_num', df_promotion, predict_results_day, \
                                    predict_cur_date, ratio_info, calendar, debug, is_solar=False)
    results = []
    for i in range(predict_periods_num * period):
        acc = 0.0
        result = 0.0
        if solar_mapd is not np.NaN:
            weight = math.exp(-solar_mapd)
            acc += weight
            result += (solar_results[i] * weight)
        if lunar_mapd is not np.NaN:
            weight = math.exp(-lunar_mapd)
            acc += weight
            result += (lunar_results[i] * weight)
        if solar_mapd_org is not np.NaN:
            weight = math.exp(-solar_mapd_org)
            acc += weight
            result += (solar_results_org[i] * weight)
        if lunar_mapd_org is not np.NaN:
            weight = math.exp(-lunar_mapd_org)
            acc += weight
            result += (lunar_results_org[i] * weight)
        result = (np.NaN if acc == 0 else result / acc)
        results.append(result)
        mapd = robust_op([solar_mapd, lunar_mapd, solar_mapd_org, lunar_mapd_org], default=np.NaN)
    return results, mapd


def lr_seasonal_forecast_base(keys, df_sku, val_key, df_promotion, predict_results_day, \
                              predict_cur_date, ratio_info, calendar, debug, is_solar=True):
    dt_key = 'order_date'
    id_key = 'sku_id'
    target = 'target'
    span = predict_periods_num * period
    results = [np.NaN] * span
    mapd = np.NaN
    if (not apply_seasonal(keys)) or df_sku[val_key].mean() <= 2.0 or df_sku.shape[0] < 30:
        return results, mapd
    new_sku = (df_sku.shape[0] < 120)
    df = lr_seasonal_features(keys, df_sku, val_key, df_promotion, predict_results_day, \
                              predict_cur_date, ratio_info, calendar, debug)
    if df is None:
        return results, mapd
    lag = 2
    try:
        if new_sku:
            all_features = ['solar_ratio']
        else:
            all_features = ['holiday_ratio', 'brand_holiday_ratio', \
                            'weekday_ratio', 'discount_ratio']
            is_seasonal = is_seasonal_category(keys, ratio_info)
            if (not is_seasonal) or (keys[keys_columns.index('item_first_cate_cd')] in ['1315']):
                all_features.extend(['feature_last_%i_value' % i for i in range(1, lag + 1)])
            if is_solar:
                all_features.extend(['solar_ratio', 'brand_solar_ratio'])
            else:
                all_features.extend(['lunar_ratio', 'brand_lunar_ratio'])
        features = []
        for col in all_features:
            if col in df.columns:
                features.append(col)
        # print('features:',features)
        fm = '{0} ~ {1} - 1'.format(target, ' + '.join(features))
        df_train = df.loc[(df[dt_key] < predict_cur_date), :]
        df_test = df.loc[df[dt_key] >= predict_cur_date, :]
        est = smf.ols(formula=fm, data=df_train).fit_regularized(method='elastic_net', alpha=2.0, L1_wt=0.1)
        fit_results = list(est.predict(df_test))
        train_results = [max(0.0, r) for r in list(est.predict(df_train))]
        if df_train.shape[0] >= 370:
            df_train['predict1'] = train_results
            df_test['predict1'] = fit_results
            features = ['feature_last_year', 'predict1', 'weekday_ratio']
            fm = '{0} ~ {1}-1'.format(target, ' + '.join(features))
            df_train = df_train[df_train[dt_key] >= predict_cur_date - timedelta(42)]
            est = smf.ols(formula=fm, data=df_train).fit_regularized(method='elastic_net', alpha=2.0, L1_wt=0.1)
            train_results = list(est.predict(df_train))
            fit_mapd = compute_mapd(np.array(df_train[target].tolist()), np.array(train_results))
            fit_results = [max(0.0, r) for r in list(est.predict(df_test))]
        else:
            fit_mapd = compute_mapd(np.array(df_train[target].tolist()), np.array(train_results))
        fit_mean = max(0.0, np.nanmean(fit_results))
        if (fit_mapd >= 3.0):
            results = [(fit_mean if (np.isnan(r) or r < 0.0) else r) for r in fit_results]
            mapd = fit_mapd
        else:
            results = fit_results
            mapd = fit_mapd
    except:
        pass
    if len(results) < span:
        results = np.append(results, [np.NaN] * (span - len(results)))
    return results, mapd


def tsfresh_seasonal_forecast(keys, df_sku, df_promotion, predict_results_day, \
                              predict_cur_date, ratio_info, calendar, debug):
    dt_key = 'order_date'
    val_key = 'original_sale_num'
    id_key = 'sku_id'
    span = predict_periods_num * period
    results = [np.NaN] * span
    mapd = np.NaN
    if (not apply_seasonal(keys)) or df_sku[val_key].mean() <= 3.0 or df_sku.shape[0] < 50:
        return results, mapd
    df = tsfresh_features(keys, df_sku, df_promotion, predict_results_day, \
                          predict_cur_date, ratio_info, calendar, debug)
    if df is None:
        return results, mapd
    df_train = df.loc[df[dt_key] < predict_cur_date, :]
    df_test = df.loc[df[dt_key] >= predict_cur_date, :]
    seasonal_features = ['year', 'month', 'lunar_week', 'solar_week', 'holiday', 'weekday', 'solar_ratio', \
                         'lunar_ratio', 'brand_solar_ratio', 'brand_lunar_ratio', 'holiday_ratio', \
                         'brand_holiday_ratio', 'synthetic_discount']
    features = []
    for col in seasonal_features:
        if col in df.columns:
            features.append(col)
    try:
        results, mapd = train_model_lgb(df_train, df_test, features, 'target')
        if len(results) < period * predict_periods_num:
            results = np.append(results, [np.NaN] * (period * predict_periods_num - len(results)))
    except:
        debug_print('seasonal_forecast_error: {0}'.format(keys), debug)
    return results, mapd


def tsfresh_forecast_incre(keys, df_sku, features, val_key, dt_key,
                           id_key, predict_cur_date, days_num,
                           debug):
    df_sku['pre'] = df_sku[val_key]
    for i in range(days_num - 1):
        df_sku.loc[:, 'pre'] = df_sku['pre'].shift(-1)
        df_sku.loc[:, val_key] = df_sku[val_key] + df_sku['pre']
    df_train = df.loc[df[dt_key] < predict_cur_date - timedelta(days_num), :]
    df_test = df.loc[df[dt_key] == predict_cur_date, :]
    result = np.NaN
    if True:  # try:
        results, mapd = train_model_lgb(df_train, df_test, features, 'target')
        result = results[0]
    # except:
    #    debug_print('tsfresh_forecast_error: {0}'.format(keys),debug)
    return result


def tsfresh_forecast(keys, df_sku, df_promotion, predict_results_day, \
                     predict_cur_date, ratio_info, calendar, debug):
    dt_key = 'order_date'
    val_key = 'original_sale_num'
    id_key = 'sku_id'
    span = predict_periods_num * period
    results = [np.NaN] * span
    mapd = np.NaN
    if (not apply_tsfresh(keys)) \
            or df_sku[val_key].mean() <= tsfresh_sale_threshold(keys) \
            or df_sku.shape[0] < 70:
        return results, mapd
    df = tsfresh_features(keys, df_sku, df_promotion, predict_results_day, \
                          predict_cur_date, ratio_info, calendar, debug)
    if df is None:
        return results, mapd
    df_train = df.loc[df[dt_key] < predict_cur_date, :]
    df_test = df.loc[df[dt_key] >= predict_cur_date, :]
    features = df.columns.tolist()
    features.remove(dt_key)
    features.remove('target')
    try:
        results, mapd = train_model_lgb(df_train, df_test, features, 'target')
        if len(results) < period * predict_periods_num:
            results = np.append(results, [np.NaN] * (period * predict_periods_num - len(results)))
        '''
        days_num = 14
        incre_result = tsfresh_forecast_incre(keys,df,features,'target',dt_key,
                           id_key,predict_cur_date,days_num,
                           debug)
        incre_result /= days_num
        org_result = np.nanmean(results[:days_num])
        for i in range(days_num):
            results[i] = (result[i] * incre_result / (1.0 if org_result == 0.0 else org_result))
        '''
    except:
        debug_print('tsfresh_forecast_error: {0}'.format(keys), debug)
    return results, mapd


def sale_flat(record, debug=False):
    df_agg = None
    for items in record[1]:
        identity = items[timeseries_filtered_index('identity')]
        sku_id = int(identity.split('#')[0])
        dc_id = int(identity.split('#')[1])
        start_date = datetime.strptime(items[timeseries_filtered_index('start_date')], '%Y-%m-%d')
        num_days = int(items[timeseries_filtered_index('len')])
        series = [try_parse(l) for l in
                  items[timeseries_filtered_index('y')].replace('[', '').replace(']', '').split(',')]
        end_date = start_date + timedelta(num_days - 1)
        date_series = pd.date_range(start_date, end_date)
        data_key = items[timeseries_filtered_index('data_key')]
        df_tmp = pd.DataFrame([[x, y] for x, y in zip(date_series, series)], \
                              columns=['order_date', data_key_columns_map[data_key]])
        df_tmp['sku_id'] = sku_id
        df_tmp['dc_id'] = dc_id
        if df_agg is None:
            df_agg = df_tmp
        else:
            df_agg = pd.merge(df_agg, df_tmp, left_on=['sku_id', 'dc_id', 'order_date'], \
                              right_on=['sku_id', 'dc_id', 'order_date'], how='inner')
    df_agg = df_agg[order_features]
    try:
        df_agg.loc['avg_jd_unit_price'] = df_agg['avg_jd_unit_price'].interpolate(method='nearest')
        df_agg['avg_sale_unit_price'].fillna(df_agg['avg_jd_unit_price'], inplace=True)
    except:
        pass
    values = record[0].split('#')
    for key in ['item_third_cate_cd', 'brand_code']:
        df_agg[key] = values[keys_columns.index(key)]
    return df_agg


def sale_info_local(record, calendar_info, debug=False):
    values = record[0].split('#')
    sales = pd.concat(record[1])
    cut_off_dt = datetime(2015, 1, 1)
    sales = sales.loc[sales['order_date'] >= cut_off_dt, :]
    df = pd.merge(sales, calendar_info.value, left_on='order_date', right_on='order_date')
    return df.values.tolist()


category_ratio_columns = ['key_name', 'key', 'ratio']
ratio_groupkeys = [['dc_id', 'item_third_cate_cd', 'solar_week'], \
                   ['dc_id', 'item_third_cate_cd', 'lunar_week'], \
                   ['dc_id', 'item_third_cate_cd', 'brand_code', 'solar_week'], \
                   ['dc_id', 'item_third_cate_cd', 'brand_code', 'lunar_week']]


def sale_ratio_category(record, calendar_info, debug=False):
    values = record[0].split('#')
    sales = pd.concat(record[1])
    cut_off_dt = datetime(2016, 1, 1)
    sales = sales.loc[sales['order_date'] >= cut_off_dt, :]
    sales = pd.merge(sales, calendar_info.value, left_on='order_date', right_on='order_date')
    sales['sale_num'].fillna(0.0, inplace=True)
    cols = []
    df_list = []
    try:
        for keys in ratio_groupkeys:
            df = sales[keys + ['sale_num']].groupby(keys).mean().reset_index()
            df['key_name'] = '#'.join(keys)
            df['key'] = df[keys].apply(lambda x: '#'.join([str(l) for l in x]), axis=1)
            df['ratio'] = df['sale_num']
            df = df[category_ratio_columns].drop_duplicates()
            df_list.append(df)
        ratio = pd.concat(df_list)
        return ratio.values.tolist()
    except:
        return []


def get_ratio_info(sc, path):
    ratio_all = sc.textFile(path) \
        .map(lambda line: line.split(',')) \
        .cache()
    keys = ratio_groupkeys[0]
    key_name = '#'.join(keys)
    ratio_solar = ratio_all.filter(lambda line: line[category_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: (line[1].split('#')[1], [('solar_' + line[1].split('#')[0], \
                                                    line[1].split('#')[-1], \
                                                    line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    keys = ratio_groupkeys[1]
    key_name = '#'.join(keys)
    ratio_lunar = ratio_all.filter(lambda line: line[category_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: (line[1].split('#')[1], [('lunar_' + line[1].split('#')[0], \
                                                    line[1].split('#')[-1], \
                                                    line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    keys = holiday_ratio_groupkeys[0]
    key_name = '#'.join(keys)
    holiday_ratio = ratio_all.filter(lambda line: line[holiday_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: (line[1].split('#')[1], [('holiday_' + line[1].split('#')[0], \
                                                    line[1].split('#')[-1], \
                                                    line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    ratio_cate = ratio_lunar.join(ratio_solar) \
        .map(lambda line: (line[0], line[1][0] + line[1][1])) \
        .join(holiday_ratio) \
        .map(lambda line: (line[0], line[1][0] + line[1][1]))

    keys = ratio_groupkeys[2]
    key_name = '#'.join(keys)
    ratio_brand_solar = ratio_all.filter(lambda line: line[category_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: ('#'.join(line[1].split('#')[1:-1]), \
                           [('brand_solar_' + line[1].split('#')[0], \
                             line[1].split('#')[-1], \
                             line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    keys = ratio_groupkeys[3]
    key_name = '#'.join(keys)
    ratio_brand_lunar \
        = ratio_all.filter(lambda line: line[category_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: ('#'.join(line[1].split('#')[1:-1]), \
                           [('brand_lunar_' + line[1].split('#')[0], \
                             line[1].split('#')[-1], \
                             line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    keys = holiday_ratio_groupkeys[1]
    key_name = '#'.join(keys)
    holiday_ratio_brand \
        = ratio_all.filter(lambda line: line[holiday_ratio_columns.index('key_name')] == key_name) \
        .map(lambda line: ('#'.join(line[1].split('#')[1:-1]), \
                           [('brand_holiday_' + line[1].split('#')[0], \
                             line[1].split('#')[-1], \
                             line[2])])) \
        .reduceByKey(lambda x, y: x + y)

    ratio_final = ratio_brand_lunar.join(ratio_brand_solar) \
        .map(lambda line: (line[0], line[1][0] + line[1][1])) \
        .join(holiday_ratio_brand) \
        .map(lambda line: (line[0], line[1][0] + line[1][1])) \
        .map(lambda line: (line[0].split('#')[0], line)) \
        .join(ratio_cate) \
        .map(lambda line: (line[1][0][0], line[1][0][1] + line[1][1])) \
        .cache()
    ratio_all.unpersist()
    ratio_solar.unpersist()
    ratio_lunar.unpersist()
    holiday_ratio.unpersist()
    ratio_cate.unpersist()
    ratio_brand_solar.unpersist()
    ratio_brand_lunar.unpersist()
    holiday_ratio_brand.unpersist()
    return ratio_final


holiday_ratio_columns = ['key_name', 'key', 'ratio']
holiday_ratio_groupkeys = [['dc_id', 'item_third_cate_cd', 'holiday'], \
                           ['dc_id', 'item_third_cate_cd', 'brand_code', 'holiday']]


def sale_ratio_holiday(record, calendar_info, debug=False):
    holidays = flatten_holiday()
    values = record[0].split('#')
    sales = pd.concat(record[1])
    cut_off_dt = datetime(2016, 1, 1)
    sales = sales.loc[sales['order_date'] >= cut_off_dt, :]
    sales = pd.merge(sales, holidays, left_on='order_date', right_on='order_date', how='left')
    sales['holiday'].fillna(0, inplace=True)
    sales['sale_num'].fillna(0, inpliace=True)
    cols = []
    df_list = []
    try:
        for keys in holiday_ratio_groupkeys:
            df = sales[keys + ['sale_num']].groupby(keys).mean().reset_index()
            df['key_name'] = '#'.join(keys)
            df['key'] = df[keys].apply(lambda x: '#'.join([str(l) for l in x]), axis=1)
            df['ratio'] = df['sale_num']
            df = df[holiday_ratio_columns].drop_duplicates()
            df_list.append(df)
        ratio = pd.concat(df_list)
        return ratio.values.tolist()
    except:
        return []


fc_parameters = {'value': {'agg_autocorrelation': [{'f_agg': 'median'},
                                                   {'f_agg': 'var'},
                                                   {'f_agg': 'mean'}],
                           'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},
                                                {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'},
                                                {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}],
                           'ar_coefficient': [{'coeff': 2, 'k': 10}],
                           'autocorrelation': [{'lag': 7}],
                           'change_quantiles': [{'f_agg': 'mean',
                                                 'isabs': False,
                                                 'qh': 1.0,
                                                 'ql': 0.6}]}}


# ==========================================================================
#  added by wusai 2018/12/20
# ===========================================================================
def chinese_new_year_correspond(df, spring_festivals, predict_cur_date, debug):
    """
        春节节假日的阳历对齐，以当前年的春节对应阳历为主。
    """
    key = False
    predict_range = pd.date_range(start=predict_cur_date, periods=91, freq='D')
    for spring_festival in spring_festivals['ds']:
        if spring_festival in predict_range:
            key = True
            this_year = str(spring_festival.year)
    if key == False:
        return df
    debug_print ('corresponding spring festival date with this year', debug)
    debug_print ('this year is ' + this_year, debug)
    # 时间格式转换
    df['order_date'] = pd.to_datetime(df['order_date'])
    spring_festivals['ds'] = pd.to_datetime(spring_festivals['ds'])
    # ==============================================================
    # 获取今年的春节日期
    # ==============================================================
    sf_date = pd.to_datetime(spring_festivals.loc[spring_festivals['ds'].dt.year == int(this_year), 'ds'].values[0])
    spring_festivals['ds'] = pd.to_datetime(spring_festivals['ds'])
    # =============================================================
    # 计算出需要对齐的历史春节和预测春节的相对距离和每个春节实际的时间窗口
    # =============================================================
    his_yrs_length = 1
    backward_days = 14
    correspond_window = 21
    his_new_year = []
    move_step = []
    his_year_note_list = []
    for i in range(1, his_yrs_length + 1):
        past_yr = str(int(this_year) - i)
        his_year_date_start = 'new_year' + past_yr  # new_year2018
        Spring_Festival = pd.to_datetime(spring_festivals.loc[spring_festivals['ds'].dt.year == int(past_yr), 'ds'].values[0])   # 过去的春节日期 例如：2018-02-16
        adjusted_Spring = this_year + '-' + str(Spring_Festival.month) + '-' + str(Spring_Festival.day) # 调整后的春节日期 #例如： 2018-02-05
        his_year_date_start = Spring_Festival + pd.DateOffset(days=-backward_days)  #春节窗口开始
        his_year_date_list = pd.date_range(start=his_year_date_start, periods=int(correspond_window),
                                           freq='D')  #春节窗口结束
        his_new_year.extend(his_year_date_list) # 需要调整春节窗口 list of date 所有日期
        move_step_tmp = (sf_date - pd.to_datetime(adjusted_Spring)).days # 需要调整的天数差距
        move_step.append(move_step_tmp) #记录下来
        his_year_note_list.append(int(past_yr)) #需要调整的春节年份
    debug_print(his_year_note_list, debug)
    # df_tmp_fix: 原始dataframe中所有春节窗口内的部分
    df_tmp_fix = df[df['order_date'].isin(his_new_year)]
    # 移动方向记录
    move_step_sign = np.sign(move_step)
    # ==============================================================================
    # 使用上面得到的时间范围开始调整每年的春节，每年春节前后窗口内的阳历日期都和预测目标阳历日期一致
    # ==============================================================================
    df_tmp_fix['year'] = df_tmp_fix['order_date'].apply(lambda x: x.year)
    df_tmp_fix['new_date'] = np.nan
    for i in range(his_yrs_length):  # 移动历史春节日期，使其阳历日期与当前年的春节阳历日期一致。
        # 'new_date' 对齐后日期 'order_date'实际对齐日期
        df_tmp_fix.loc[df_tmp_fix.year == his_year_note_list[i], 'new_date'] = \
            df_tmp_fix[df_tmp_fix.year == his_year_note_list[i]]['order_date'].apply(
                lambda x: x + pd.DateOffset(days=move_step[i]))
    # =============================================================================
    # 补齐剩余的受影响的日期 (移动方式替换)
    # =============================================================================
    # 需要补齐的时间部分（在春节窗口之外的部分需要移到另一侧的空缺补齐）
    new_move_date_his = list(set(df_tmp_fix.new_date) - set(df_tmp_fix['order_date']))
    # df_tmp_other_fix: 需要补齐的时间外的矩阵
    df_tmp_other_fix = df[df['order_date'].isin(new_move_date_his)]
    df_tmp_other_fix['year'] = df_tmp_other_fix['order_date'].apply(lambda x: x.year)
    df_tmp_other_fix['new_date'] = np.nan
    # 把每个需要补齐的年份依次对齐后补齐
    for i in range(his_yrs_length):
        windows = -(move_step_sign[i]) * float(correspond_window)
        print('---windows----')
        print(windows)
        print(df_tmp_other_fix.head())
        df_tmp_other_fix.loc[df_tmp_other_fix.year == his_year_note_list[i], 'new_date'] \
            = df_tmp_other_fix[df_tmp_other_fix.year == his_year_note_list[i]]['order_date'].apply(
            lambda x: x + pd.DateOffset(days=windows))
    # df_other: 既不在需要补齐的日期中也不再春节窗口内的部分dataframe
    df_other = df[(~df['order_date'].isin(new_move_date_his)) & (~df['order_date'].isin(his_new_year))]
    # =================================================================================
    # 还原回原dataframe的格式，但更新日期对应行记录的顺序
    # ======================================================================================
    df_tmp_fix['new_date'] = pd.to_datetime(df_tmp_fix['new_date'])
    df_tmp_other_fix['new_date'] = pd.to_datetime(df_tmp_other_fix['new_date'])
    del df_tmp_fix['order_date']
    del df_tmp_fix['year']
    del df_tmp_other_fix['order_date']
    del df_tmp_other_fix['year']
    df_tmp_fix = df_tmp_fix.rename(columns={'new_date': 'order_date'})
    df_tmp_other_fix = df_tmp_other_fix.rename(columns={'new_date': 'order_date'})
    df_new = pd.concat([df_tmp_fix, df_tmp_other_fix, df_other])  # 对齐春节节假日的新dataframe
    df_new.sort_values(by=['order_date'], inplace=True)
    return df_new

def gen_ts_sql(ts_table, filter_dc_id_list=None, filter_subquery_sql=None):

    #joint query by dc_id
    filter_dc_id_sql = ""
    if (filter_dc_id_list is not None and filter_dc_id_list != "0"):
        dc_id_str=""
        for dc_id in filter_dc_id_list.split(","):
            dc_id_str += "'" + dc_id + "',"
        dc_id_str = dc_id_str[:-1]

        filter_dc_id_sql = " and split(identity,'#')[1] in (" + dc_id_str + ")"

    # joint query by subquery
    filter_by_identity_sql = ""
    if (filter_subquery_sql is not None):
        filter_by_identity_sql = " left semi join ({0}) t on ts.identity=t.identity ".format(filter_subquery_sql)

    ts_sql = """
        select ts.* from (
        select
            identity,
            data_key,
            start_date,
            len,
            y
        from
            {ts_table}
        where
            key in('salesForecast', 'priceBeforeDiscount', 'priceAfterDiscount', 'stockQuantity')
            {filter_dc_id_sql}
        ) ts
        {filter_by_identity_sql}

    """.format(ts_table=ts_table, filter_dc_id_sql=filter_dc_id_sql, filter_by_identity_sql=filter_by_identity_sql)
    return ts_sql