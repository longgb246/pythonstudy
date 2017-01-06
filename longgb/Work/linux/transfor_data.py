#-*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

# path = r'/home/cmo_ipc/Allocation_shell/datasets'
path_a = r'D:\Lgb\ReadData\02_allocation\000000_0'
data_test = pd.read_table(path_a, sep='\001', header=None)

import pickle
path_b = r'D:\Lgb\ReadData\02_allocation\aa_test.pkl'
output = open(path_b, 'wb')
pickle.dump(data_test, output)
output.close()

import cPickle as p
output = open(path_b, 'wb')
data2 = pickle.load(output)
output.close()



