#-*- coding:utf-8 -*-
import os
import pandas as pd

def uppath(n=1):
    if n == 0:
        return os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), (os.pardir + os.sep) * (n - 1) + os.pardir))

print uppath(4)




data_origin = pd.DataFrame()
data_origin["supp_brevity_cd"].drop_duplicates().count()
data_origin["pur_bill_id"].count()

