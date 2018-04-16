#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import os
import cPickle


def plotTimeSeries(data, dt, target, layer_col=None, layer=None, save_name=None, save_path=None):
    '''
    Plot Time Series
    :param data: pd.DataFrame,
    :param dt: string, dt name
    :param target: string, dt name
    :param layer_col: string, layer name
    :param layer: list, layer value
    :param save_name: string, save name
    :param save_path: string, save path
    '''
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    if layer is not None:
        for each_layer in layer:
            sub_data = data[data[layer_col]==each_layer].sort_values(dt)
            plot_x = map(lambda m: datetime.datetime.utcfromtimestamp(m.astype(float) * 1e-9), pd.to_datetime(sub_data[dt]).values)
            ax.plot(plot_x, sub_data[target], '-', label='{0}_{1}'.format(layer_col, each_layer))
    else:
        data = data.sort_values(dt)
        plot_x = map(lambda m: datetime.datetime.utcfromtimestamp(m.astype(float) * 1e-9), pd.to_datetime(data[dt]).values)
        ax.plot(plot_x, data[target], '-', label=target)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path + os.sep + save_name)
    return fig


def dateCalculate(start_date, cal_date=0):
    '''
    From the start date to a certain direction to get a date list.
    :param start_date: Start date to calculate
    :param cal_date: From the start date to a certain direction.
    :return: list
    (example)
        In[1]: myTools.dateCalculate('2017-03-04', 3)
        Out[1]: ['2017-03-04', '2017-03-05', '2017-03-06', '2017-03-07']
        In[2]: myTools.dateCalculate('2017-03-04', 0)
        Out[2]: ['2017-03-04']
        In[3]: myTools.dateCalculate('2017-03-04', -3)
        Out[3]: ['2017-03-01', '2017-03-02', '2017-03-03', '2017-03-04']
    '''
    start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = start_date_dt + datetime.timedelta(cal_date)
    min_date = min(start_date_dt, end_date_dt)
    max_date = max(start_date_dt, end_date_dt)
    date_range = map(lambda x: (min_date + datetime.timedelta(x)).strftime('%Y-%m-%d'),range((max_date - min_date).days + 1))
    return date_range


def getTestData():
    n_data = 100
    x = np.arange(n_data)
    x_axis = dateCalculate('2017-01-02', cal_date=n_data-1)
    y1 = (np.sin(x*1.0/10)*3).tolist()
    data_t1 = pd.DataFrame(zip(x_axis, y1), columns=['dt', 'value'])
    data_t1['layer'] = 0
    y2 = (np.cos(x*1.0/10)*3+0.5).tolist()
    data_t2 = pd.DataFrame(zip(x_axis, y2), columns=['dt', 'value'])
    data_t2['layer'] = 1
    data_t = pd.concat([data_t1, data_t2])
    return data_t


if __name__ == '__main__':
    data_t = getTestData()
    # data = getTestData()
    fig1 = plotTimeSeries(data_t, 'dt', 'value', layer_col='layer', layer=[0, 1])
    with open(r'E:\fig1rrrr.pkl', 'r') as f:
        cPickle.dump(fig1, f)


# with open(r'E:\fig1.pkl', 'r') as f:
#     fig1 = cPickle.load(f)
# fig1.savefig(r'E:\fig1.png')

