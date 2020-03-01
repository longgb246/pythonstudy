# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/02/16
"""
Usage Of 'ncp_ljj_kaggle.py' : 
"""

'''
数据来源：https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset
'''


from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot, plot_plotly
from fbprophet.diagnostics import cross_validation, performance_metrics
import os
import warnings
from fbprophet import Prophet
from datetime import datetime, date, timedelta
from plotly.subplots import make_subplots
from plotly import subplots
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
plt.style.use('ggplot')


warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv(
    '/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
df.head(5)

df[df['Province/State'].isnull()]

fig = px.bar(df, x='Date', y='Confirmed', hover_data=[
             'Province/State', 'Deaths', 'Recovered'], color='Country')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text='Confirmed bar plot for each country',
                        font=dict(family='Arial',
                                  size=30,
                                  color='rgb(37,37,37)'),
                        showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()

fig = px.bar(df.loc[dataset['Country'] == 'Mainland China'], x='Date', y='Confirmed', hover_data=[
             'Province/State', 'Deaths', 'Recovered'], color='Province/State')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text='Confirmed bar plot for Mainland China',
                        font=dict(family='Arial',
                                  size=30,
                                  color='rgb(37,37,37)'),
                        showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()

confirmed_training_dataset = pd.DataFrame(dataset[dataset.Country == 'China'].groupby(
    'Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})
confirmed_training_dataset.head()


prophet = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive')
prophet.fit(confirmed_training_dataset)
future = prophet.make_future_dataframe(periods=7)
confirmed_forecast = prophet.predict(future)

fig = plot_plotly(prophet, confirmed_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text='确诊人数预测',
                        font=dict(family='Arial',
                                  size=30,
                                  color='rgb(37,37,37)'),
                        showarrow=False))
fig.update_layout(annotations=annotations)

prophet = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive')
prophet.fit(confirmed_training_dataset)
future = prophet.make_future_dataframe(periods=7)
confirmed_forecast_2 = prophet.predict(future)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


max_date = prophet.ds.max()
y_true = prophet.y.values
y_pred_daily = confirmed_forecast.loc[confirmed_forecast['ds']
                                      <= max_date].yhat.values
y_pred_daily_2 = confirmed_forecast_2.loc[confirmed_forecast_2['ds']
                                          <= max_date].yhat.values

print('包含日季节性 MAPE: {}'.format(
    mean_absolute_percentage_error(y_true, y_pred_daily)))
print('不包含日季节性 MAPE: {}'.format(
    mean_absolute_percentage_error(y_true, y_pred_daily_2)))


# https://zhuanlan.zhihu.com/p/52330017
# https://www.kaggle.com/shubhamai/coronavirus-eda-future-predictions
# https://www.kaggle.com/parulpandey/wuhan-coronavirus-a-geographical-analysis
