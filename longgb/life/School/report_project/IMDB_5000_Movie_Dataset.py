#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# =                             （1）读入数据                                 =
# ============================================================================
read_path = r'D:\Lgb\Self\project2\imdb-5000-movie-dataset'
df = pd.read_csv(read_path + os.sep + 'movie_metadata.csv')
df.head()

keys = np.sort(df.columns)
maxKeyLength = max(map(lambda x: len(x), keys))
print('number of keys: %s\n\n' % len(keys))
for i, key in enumerate(keys):
    print ('•{:%s}' % (maxKeyLength)).format(key),
    if i % 4 == 3:
        print('\n')



