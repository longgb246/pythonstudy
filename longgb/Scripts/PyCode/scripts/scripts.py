# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : longguangbin@163.com
  Date    : 2018/9/11
  Usage   : 
"""

import pandas as pd

tmp_df = pd.DataFrame([['2018-01-01 14:01:00', 3],
                       ['2018-01-01 14:02:00', 4],
                       ['2018-01-01 14:03:00', 6],
                       ['2018-01-01 14:04:00', 32],
                       ['2018-01-01 14:05:00', 10],
                       ['2018-01-01 14:06:00', 9],
                       ['2018-01-01 14:07:00', 8],
                       ], columns=['dt', 'globalRadiation'])
tmp_df = tmp_df.set_index(['dt'])
# reset index, only set auto series as index
tmp_df = tmp_df.reset_index()
tmp_df['before_min'] = tmp_df['globalRadiation'].shift(5)
tmp_df = tmp_df.fillna(0)
tmp_df['delta'] = tmp_df['globalRadiation'] - tmp_df['before_min']
