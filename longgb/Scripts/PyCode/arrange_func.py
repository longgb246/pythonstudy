# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/4/1
"""  
Usage Of 'arrange_func.py' : 
"""

# ==================================================
# =     Pandas
# ==================================================
import pandas as pd

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 80)  # 150 / 180
pd.set_option('display.max_columns', 40)


def cross_join(a_df, b_df):
    """ pandas dataframe cross join. """
    a_df['tmp_cross_join'] = '1'
    b_df['tmp_cross_join'] = '1'
    c_df = a_df.merge(b_df, on=['tmp_cross_join'])
    c_df = c_df.drop(['tmp_cross_join'], axis=1)
    return c_df
