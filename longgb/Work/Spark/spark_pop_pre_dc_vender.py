#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'zhangjianshen'

import os
import sys
import datetime as dt
import traceback

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext, HiveContext, Column, Row
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from scipy.optimize import fmin_l_bfgs_b, minimize
#######################预测调用函数，holt_winter,additive#####################
def moving_average(a,n=3,n_preds=28) :
    b=[]
    for i in range(len(a)+n_preds):
        if i==0:
            pre=a[i]
            b.append(round(pre))
        elif 0<i<n:
            pre=sum(a[:i])/i
            b.append(round(pre))
        elif n<=i<len(a):
            pre=sum(a[(i-n):i])/n
            b.append(round(pre))
        else:
            pre=sum(b[-n:])/n
            b.append(round(pre))
    return b


def moving_average_new(time_series,n_preds=28):
    history_len=len(time_series)
    mapd,n=moving_average_train(time_series)
    result_series=moving_average_predict(time_series,n,n_preds)
    return result_series,mapd


def compute_mapd(predict_list,real_list):
    n=len(real_list)
    weighted_mapd = 0.0
    for real, predict in zip(real_list, predict_list):
        if real==0:
            weighted_mapd += predict
        else:
            weighted_mapd += abs((real - predict) / real)
    return weighted_mapd/n

def moving_average_train(time_series):
    mapd = np.NaN
    n_param = np.NaN
    for n in range(1,20,1):
        result_series=moving_average(time_series, n,0)
        cur_mapd=compute_mapd(time_series,result_series)
        if mapd==None or cur_mapd < mapd:
            mapd = cur_mapd
            n_param = n
    return mapd,n_param

def moving_average_predict(time_series,n_param,n_preds):
    result_series=moving_average(time_series,n_param,n_preds)
    result_series_tmp=result_series[-n_preds:]
    result_series_predict=[]
    for i in range(len(result_series_tmp)):
        result_series_predict.append(round(float(result_series_tmp[i]),6))
    return result_series



def RMSE(params,*args):
    Y = args[0]
    type = args[1]
    rmse = 0
    if type == 'linear':
        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])
    else:
        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        if type == 'additive':
            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]
            for i in range(len(Y)):
                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])
        elif type == 'multiplicative':
            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]
            for i in range(len(Y)):
                a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y.append((a[i + 1] + b[i + 1]) * s[i + 1])
        else:
            exit('Type must be either linear, additive or multiplicative')
    rmse = (sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))**0.5
    return rmse

def additive(x, m, fc, alpha=None, beta=None, gamma=None):
    Y = x[:]
    if (alpha == None or beta == None or gamma == None):
        initial_values = [0.2, 0, 0.1]
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'additive'
        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type, m), bounds=boundaries, approx_grad=True)
        alpha, beta, gamma = parameters[0]
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] - a[0] for i in range(m)]
    y = [a[0] + b[0] + s[0]]
    rmse = 0
    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] + b[-1] + s[-m])
        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])
    rmse = (sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))**0.5
    return Y[-fc:], alpha, beta, gamma, rmse

def sales_forecast(line):
    time_series=line[6]
    mean_sales=round(sum(line[6])/len(line[6]),6)
    mean_last=round(sum(line[6][-5:])/len(line[6][-5:]),6)
    if (mean_last>2*mean_sales) and (len(time_series)>10):
        real_preds_package = additive(time_series,m=7,fc=28,alpha=0.1, beta=0.1, gamma=0.2)
        preds_list = real_preds_package[0]
    elif (mean_last<=2*mean_sales) and (len(time_series)>10):
        real_preds_package = additive(time_series,m=7,fc=28)
        preds_list = real_preds_package[0]
    else:
        real_preds_list= moving_average_new(time_series,n_preds)
        preds_list = real_preds_list[-n_preds:]
    return(line[0],line[1],line[2],line[3],preds_list)


def main(data_path,seg_date):
    conf = SparkConf().setAppName("spark_ipc_sfs_app_standard_dc_salesper")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    df = spark.read.csv(data_path,sep='\t',header = True)
    df = df.filter(df.package_dt<=seg_date).groupBy(['pop_vender_id','pop_vender_name','dc_id','package_dt']).agg(sum('ord_qtty').alias('ord_qtty'))
    sqlContext = SQLContext(sc)
    threshold_sales = df.filter(df.package_dt<=seg_date).select('pop_vender_id','pop_vender_name','dc_id','ord_qtty')
    threshold_sales = threshold_sales.select(threshold_sales.pop_vender_id,threshold_sales.pop_vender_name,threshold_sales.dc_id,
                                             threshold_sales.ord_qtty.astype(DoubleType()).alias('ord_qtty')).groupBy(['pop_vender_id','pop_vender_name','dc_id']).agg(avg('ord_qtty').alias('sales_thre'))

    df_sales1=df.join(threshold_sales,on=['pop_vender_id','pop_vender_name','dc_id'])
    df_sales2=df_sales1.select(df_sales1.pop_vender_id,df_sales1.pop_vender_name,df_sales1.dc_id,df_sales1.package_dt,
                               F.when((df_sales1.sales_thre>=5) & (df_sales1.ord_qtty<=0.4*df_sales1.sales_thre),df_sales1.sales_thre*0.4).when(
                                   (df_sales1.sales_thre>=5)&(df_sales1.ord_qtty>=2*df_sales1.sales_thre),df_sales1.sales_thre).otherwise(df_sales1.ord_qtty).alias('ord_qtty'))
    group_sales1=df_sales2.filter(df_sales2.package_dt<=seg_date).orderBy(['pop_vender_id','pop_vender_name','dc_id','package_dt'],ascending=True).groupBy(
        ['pop_vender_id','pop_vender_name','dc_id']).agg(F.collect_list('ord_qtty').alias('ord_qtty'),F.collect_list('package_dt').alias('package_dt'))
    return group_sales1

def deal_data(df):
    time_series=[0,df.dc_id,df.pop_vender_id,df.pop_vender_name,0,0,df.ord_qtty]
    print time_series
    sales_forecast(time_series)
    # return[0,0,df.pop_vender_id,df.pop_vender_name,df.package_dt]

if __name__ == "__main__":
    data_path='/user/cmo_ipc/zhangjianshen/order_qtty_20170505104541.csv'
    dealed_data = main(data_path,'2017-03-20')
    # dealed_data.foreach(deal_data)
    ########################预测######################
    n_preds=28
    # r = os.system("hadoop fs -test -e /user/cmo_ipc/forecast/result/rdc/ratio/tmp_dc_ratio_phx0901/_SUCCESS")
    r=0
    if r != 0:
        sc.stop()
        raise Exception("1")
    else:
        print ("执行spark获取商家订单预测成功！！")