#-*- coding:utf-8 -*-

## 1
## 可以测一下速度，与rdd的转换相关的另一套运行。
## 还有使用udf的spark.df
# sp.groupBy(key).agg(F.collect_list(F.col(target))).rdd.flatMap(_deal_ts)      # _deal_ts是一个函数，返回的是list

## 2
## 2 个 spark.df join之前，最好做一次这个操作，具体 bug ，回忆一下就想起来了
# pdcols = self.predict_df.columns
# self.predict_df = self.predict_df.rdd.toDF(pdcols)

## 3
## 使用 Window、collect_list、struct 的联合使用
# windowspec_r = Window.orderBy(self.ds).partitionBy(self.key).rowsBetween(Window.currentRow,self.npred-1)
# his_list_df = self.history_df.select(self.key,self.ds,F.udf(sorter)(F.collect_list(F.struct(self.ds,self.target)).over(windowspec_r)).alias(self.ban_str+self.target))

## 4
## 选择 rank 是 1 的，按照某个排序的第一条数据
# window = Window.partitionBy(key).orderBy(F.col(ds))
# FW_data.select('*', F.rank().over(window).alias('_valid_train_rank')).filter(F.col('_valid_train_rank') <= 1)

## 过滤条件
# valid_train_df.filter(F.col(key).like('%forecast%'))

## Hive: 日期转星期函数
# select date_format('2018-04-08' ,'u')

## Hive: 从 HDFS load 数据到 Hive
# "hive -e \"load data inpath \'"+OUTPUT_URL+"\' overwrite into table "+'app.app_saas_sfs_rst'+" partition(tenant_id="+'3'+",dt=\'"+'2018-04-10'+"\');\""

# partitionBy(forecast_len, partitionFunc=lambda x : part_dict[x])

# param_dict = {'aa':11,'bb':22,'cc':33,'dd':44}
# '{aa},{bb},{cc}'.format(**param_dict)
# param_dict = {'aa':11,'bb':22,'cc':33,'dd':44}
# '{aa},{bb},{cc}'.format(aa=11,bb=22,cc=3)

