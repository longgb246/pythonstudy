# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/6
"""  
Usage Of 'local_test_1' : 
"""

import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 150)  # 150
pd.set_option('display.max_columns', 5)
# pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 20)


# --------------------------------------------------------
# Tools
def group_analysis(data, col):
    return data.groupby(col)[col].count()


def list_reduce(list_v, reduce_v):
    list_v = copy.deepcopy(list_v)
    for r_v in reduce_v:
        list_v.remove(r_v)
    return list_v


# --------------------------------------------------------
# Model
from pyramid.arima import auto_arima
from pyramid.arima.utils import ndiffs
from pyramid.arima.utils import nsdiffs

from statsmodels.tsa.holtwinters import Holt

from fbprophet import Prophet

import xgboost as xgb
import lightgbm as lgb


class BaseModel(object):

    def __init__(self):
        self.model_name = None
        self.model = None
        self.pred = None
        self.org_data = None
        self.pre_len = None

    def _get_model(self, kwargs):
        self.model_name = kwargs.get('model_name')
        self.model = kwargs.get('model')
        self.pred = kwargs.get('pred')
        self.org_data = kwargs.get('org_data')
        self.pre_len = kwargs.get('pre_len')

    def plot_pre(self):
        data = self.org_data
        pre_len = self.pre_len
        output = self.pred

        real_index = range(len(data))
        pre_index = range(len(data), len(data) + pre_len)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(real_index, data, label='sale')
        ax.plot(pre_index, output, label='pre_sale')
        ax.set_title('Time Series Model', fontsize=18)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.55))
        plt.subplots_adjust(right=0.85)
        plt.show()


class TSModel(BaseModel):

    def __init__(self):
        super(TSModel, self).__init__()

    def autoarima(self, data, pre_len=7):
        D_f = nsdiffs(data, m=3, max_D=5, test='ch')
        d_f = ndiffs(data, alpha=0.05, test='kpss', max_d=5)
        if len(data) <= 30:
            seasonal = False
        else:
            seasonal = True
        try:
            stepwise_fit = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=3, m=12,
                                      start_P=0, seasonal=seasonal, d=d_f, D=D_f, trace=False,
                                      error_action='ignore',  # don't want to know if an order does not work
                                      suppress_warnings=True,  # don't want convergence warnings
                                      stepwise=True)  # set to stepwise
        except:
            stepwise_fit = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=0, m=12,
                                      start_P=0, seasonal=False, d=0, D=0, trace=False,
                                      error_action='ignore',  # don't want to know if an order does not work
                                      suppress_warnings=True,  # don't want convergence warnings
                                      stepwise=True)  # set to stepwise
        output = stepwise_fit.predict(n_periods=pre_len).tolist()

        self._get_model({'model_name': 'autoarima',
                         'model': stepwise_fit,
                         'pred': output,
                         'org_data': data,
                         'pre_len': pre_len})

        return output

    def cap_prophet(self, data, pre_len=7, return_all=False):
        df = data.copy()

        df.columns = ['ds', 'y']
        pre_date = max(df['ds'])
        max_value = np.max(df['y']) * 1.2
        df['cap'] = max_value
        df['floor'] = 0

        if len(df) == 1:
            return [df['y'].values[0]] * pre_len

        if max_value:
            m = Prophet(growth='logistic')
        else:
            m = Prophet()
        try:
            m.fit(df)
        except:
            raise Exception(df)

        future = m.make_future_dataframe(periods=pre_len)
        future['cap'] = np.max(df['y']) * 1.2
        future['floor'] = 0
        forecast = m.predict(future)

        if return_all:
            return forecast

        pre = forecast.loc[forecast['ds'] > pre_date, ['ds', 'yhat']]
        pre.index = range(len(pre))
        pre_list = pre['yhat'].values.tolist()

        self._get_model({'model_name': 'cap_prophet',
                         'model': m,
                         'pred': pre_list,
                         'org_data': data,
                         'pre_len': pre_len})

        return pre_list

    def hw(self, data, pre_len=7):
        m = Holt(data).fit()
        out = m.predict(pre_len)[-pre_len:].tolist()

        self._get_model({'model_name': 'holt_winter',
                         'model': m,
                         'pred': out,
                         'org_data': data,
                         'pre_len': pre_len})
        return out

    def arima(self):
        pass

    def sarima(self):
        pass


class REGModel(BaseModel):

    def __init__(self):
        super(REGModel, self).__init__()

    def xgboost(self, train_x, train_y, test_x):
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x)
        pre_len = len(test_x)

        params = {'booster': 'gbtree',
                  'objective': 'reg:gamma',
                  # 'eval_metric': 'auc',
                  'max_depth': 4,
                  # 'lambda': 10,
                  'subsample': 0.75,
                  'colsample_bytree': 0.75,
                  'min_child_weight': 2,
                  'eta': 0.025,
                  # 'seed': 0,
                  'learning_rate': 0.1,
                  'nthread': 8,
                  'silent': 1
                  }

        bst = xgb.train(params, dtrain, num_boost_round=100)
        y_pred = bst.predict(dtest)

        self._get_model({'model_name': 'xgboost',
                         'model': bst,
                         'pred': y_pred,
                         'org_data': train_x,
                         'pre_len': pre_len})

        return y_pred

    def lightgbm(self, train_x, train_y, test_x, test_y):
        # print('Start training...')
        # # 创建模型，训练模型
        # gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
        # gbm.fit(train_x, train_y, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
        #
        # print('Start predicting...')
        # # 测试机预测
        # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

        # 创建成lgb特征的数据集格式
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
        pre_len = len(test_x)

        # 将参数写成字典下形式
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }

        # 训练 cv and train
        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5,
                        verbose_eval=False)

        # 预测数据集
        y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)

        self._get_model({'model_name': 'lightgbm',
                         'model': gbm,
                         'pred': y_pred,
                         'org_data': train_x,
                         'pre_len': pre_len})

        # 评估模型
        # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
        return y_pred


# --------------------------------------------------------
# Use
def get_data():
    path = r'/Users/longguangbin/Work/scripts/local_test/anta_online'

    train_df = pd.read_table(path + os.sep + 'train_df.tsv')
    target_df = pd.read_table(path + os.sep + 'target_df.tsv')

    new_target_df = target_df.loc[target_df['_model_order_'] == -1, ['sku_id', 'sale_y']]

    del_cols = ['_slide__model_r_', '_slide_target_flag', 'dt', 'dynamic_dims', 'granu', '_valid_train_rank',
                'cate1', 'cate2', 'cate3', 'pre_target_dimension_id', '_pipeline__new_granularity_mark',
                '__slide_window_col_name_mm__']

    keep_cols = list_reduce(train_df.columns.tolist(), del_cols)
    new_train_df = train_df.loc[:, keep_cols]

    new_target_df = new_train_df.loc[:, ['sku_id']].merge(new_target_df, on=['sku_id'], how='left').fillna(0)

    return new_train_df, new_target_df


def main():
    train_df, target_df = get_data()

    pass


if __name__ == '__main__':
    main()
