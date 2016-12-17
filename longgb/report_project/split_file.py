#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np

# 拆分文件
split = 3
path_a = r''
file = pd.read_table(path_a + os.sep + '000000_0', sep='\001', header=None)
len_n = len(file.index)
split_line = map(int,np.linspace(0, len_n, split + 1))[:-1] + [len_n - 1]
for i in range(split):
    file_tmp = file.iloc[range(split_line[i], split_line[i+1]),:]
    file_tmp.to_csv(path_a + os.sep + '000000_0_{0}.csv'.format(i+1), index=False, header=None)

