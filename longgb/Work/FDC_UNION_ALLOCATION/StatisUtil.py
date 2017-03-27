#encoding=utf-8
import pandas as pd
import statsmodels.api as sm
import numpy as np
lowess = sm.nonparametric.lowess
class Lowess:
    @staticmethod
    def smooth(dataFrame,theta=3,frac=0.3,it=2):
        """
        Author By: Gao Yun
        Date: 2016-09-12
        LastUpdate: 2016-10-18
        input:
            dataFrame: (date Datetime,sku int,sales int)
            theta int: x times of error
            frac float: Between 0 and 1. The fraction of the data used when estimating each y-value.
            it int: The number of residual-based reweightings to perform.

        output: Dataframe(sku,sales,sales_smooth)
        tutual:   test_sku_1001177.txt --   date	    sku	    sales
                                            2016-01-01	1001177	3
            dataFrame = pd.read_csv(r"D:\tmp\data\test_sku_1001177.txt",sep="\t",header=0)
            data=[]
            dataFrame["date"]=pd.to_datetime(dataFrame["date"])
            dataFrame.sort_values(by=["sku","date"],inplace=True)
            for sku,grouped in dataFrame.groupby("sku"):
                grouped = grouped.set_index("date")
                grouped["sku"] = sku
                data.append(Lowess.smooth(grouped))
            dataFrame = pd.concat(data)
        """
        data_list=[]
        for sku,grouped in dataFrame.groupby('sku'):
            lowes = lowess(grouped["sales"],grouped.index,frac=frac,it=it)
            dataframe2 = pd.DataFrame(lowes)

            dataframe2.columns=["date","sales_smooth0"]

            dataframe2=dataframe2[["sales_smooth0"]]
            grouped.reset_index(inplace=True)
            # print dataframe2
            # # dataframe2["date"]=pd.to_datetime(dataframe2["date"])
            # dataframe2.set_index("date",inplace=True)
            dataframe3 = grouped.join(dataframe2)
            dataframe3["err"]=dataframe3["sales"].astype(float) - dataframe3["sales_smooth0"]
            dataframe3["avg"] = dataframe3["err"].mean(axis=0)
            dataframe3["std"] = dataframe3["err"].std(axis=0)
            dataframe3["abnormal"] = (abs(dataframe3["err"]))>= (theta* dataframe3["std"])
            # print dataframe3[dataframe3["abnormal"]==True]#.ix[:,1:]
            # dataframe2.plot()
            # plt.show()
            dataframe3["sales_smooth"]=dataframe3.apply(lambda df: (np.ceil(df["sales_smooth0"]) if np.ceil(df["sales_smooth0"])<=np.ceil(df["sales"]) else np.ceil(df["sales"])) if df["abnormal"]==True else np.ceil(df["sales"]),axis=1)
            dataframe3["sku"]=sku
            data_list.append(dataframe3)
        dataframe = pd.concat(data_list,ignore_index=True)
        dataframe.reset_index()

        return dataframe.loc[:,["sku","date","sales","sales_smooth",]]

class EMsmooth:
    @staticmethod
    def smooth(df,theta=3,threshold=15):
        """
        input a dataframe, and output a new one with column 'sales' smoothed
        :param df: original dataframe
        :param theta: times of sigma used to identify abnormal value
        :param threshold: only higher than this can be thought as abnormal value
        :return: a new dataframe with column 'sales' smoothed
        """
        df['date'] = pd.to_datetime(df["date"])
        df = df.sort_values('date')
        dataframe = df.copy()
        dataframe['sales_smooth'] = dataframe['sales']
        data_list=[]
        for sku,grouped in dataframe.groupby('sku'):
            # print u'优化函数中，处理sku: ',sku
            grouped = grouped.sort_values('sales_smooth', ascending = False)
            grouped = grouped.reset_index()
            diff = 1000
            i = 0
            n = grouped.shape[0]
            while diff > 0:
                j = i
                avg = grouped.sales_smooth[(i+1):n].mean()
                sigma = grouped.sales_smooth[(i+1):n].std()
                if (grouped.sales_smooth[i] > avg + theta*sigma) & (grouped.sales_smooth[i] > threshold):
                    i += 1
                diff = i-j
            if i > 0: # 至少找出一个点
                grouped.sales_smooth[:(i-1)] = np.nan
                grouped = grouped.sort_values('date')
                grouped.sales_smooth = grouped.sales_smooth.interpolate() # 缺失值用插值替代
            else:
                grouped = grouped.sort_values('date')
            data_list.append(grouped)
        dataframe2 = pd.concat(data_list,ignore_index=True)
        dataframe2.sales_smooth = np.ceil(dataframe2.sales_smooth) # 向上取整
        del dataframe2['index']
        return dataframe2.loc[:,["sku","date","sales_smooth"]]


