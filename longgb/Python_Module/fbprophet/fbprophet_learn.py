#-*- coding:utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')                # 使用tk画图
import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet


abs_path = "/Users/longguangbin/Work/Codes/pythonstudy/longgb/Python_Module/fbprophet/source/prophet-master"


def simple1():
    """
    传入序列，拆分因素。
    :return:
    """
    sample_data_path = abs_path + "/examples/example_wp_log_peyton_manning.csv"
    df = pd.read_csv(sample_data_path)
    df.head()
    # fit
    m = Prophet()
    m.fit(df)
    # make future length
    future = m.make_future_dataframe(periods=365)
    future.tail()
    # predict
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # plot
    fig1 = m.plot(forecast)
    fig1.show()
    fig2 = m.plot_components(forecast)
    fig2.show()


def saturatingForecasts():
    """
    饱和值预测。我理解，有最大、最小值界限
    :return:
    """
    df = pd.read_csv(abs_path + '/examples/example_wp_log_R.csv')
    df['cap'] = 8.5

    m = Prophet(growth='logistic')
    m.fit(df)

    future = m.make_future_dataframe(periods=1826)
    future['cap'] = 8.5
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()

    df['y'] = 10 - df['y']
    df['cap'] = 6
    df['floor'] = 1.5
    future['cap'] = 6
    future['floor'] = 1.5
    m = Prophet(growth='logistic')
    m.fit(df)
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()




