#-*- coding:utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_2pics(data_tmp, name_list1, name_list2,save_path):
    plt.figure(1)
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)
    for each in name_list1:
        p1.plot(data_tmp['date_s'], data_tmp[each], linewidth=2)
    for each in name_list2:
        p2.plot(data_tmp['date_s'], data_tmp[each], linewidth=2)
    p1.set_xlabel(u"Date")
    p1.set_ylabel(u"Quantity")
    p1.set_title(u"Inv .. Sale")
    p2.set_xlabel(u"Date")
    p2.set_ylabel(u"Quantity")
    p2.set_title(u"Inv .. Sale")


read_path = r'D:\Lgb\data_sz'
save_path = r'D:\Lgb\data_sz'


if __name__ == '__main__':
    fdc_inv_compare = pd.read_csv(read_path + os.sep + 'fdc_inv_compare.csv')
    fdc_inv_compare_sku = list(set(fdc_inv_compare['sku']))
    sale_compare = pd.read_csv(read_path + os.sep + 'sale_compare.csv')
    sale_compare = list(set(sale_compare['sku']))





