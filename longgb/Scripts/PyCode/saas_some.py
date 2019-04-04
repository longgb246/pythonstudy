#-*- coding:utf-8 -*-

## 1
## 可以测一下速度，与rdd的转换相关的另一套运行。
## 还有使用udf的spark.df
# sp.groupBy(key).agg(F.collect_list(F.col(target))).rdd.flatMap(_deal_ts)      # _deal_ts是一个函数，返回的是list

## 2
## 2 个 spark.df join之前，最好做一次这个操作，具体 bug ，回忆一下就想起来了
# pdcols = self.predict_df.columns
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
# 'hive -e \'load data inpath \''+OUTPUT_URL+'\' overwrite into table '+'app.app_saas_sfs_rst'+' partition(tenant_id='+'3'+',dt=\''+'2018-04-10'+'\');\''

# partitionBy(forecast_len, partitionFunc=lambda x : part_dict[x])

# param_dict = {'aa':11,'bb':22,'cc':33,'dd':44}
# '{aa},{bb},{cc}'.format(**param_dict)
# param_dict = {'aa':11,'bb':22,'cc':33,'dd':44}
# '{aa},{bb},{cc}'.format(aa=11,bb=22,cc=3)


# def getSparkDateRangeP2(start, end):
#     if isinstance(start,str):
#         start_date_dt = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S') if len(start)>10 else datetime.datetime.strptime(start, '%Y-%m-%d')
#         end_date_dt = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') if len(end)>10 else datetime.datetime.strptime(end, '%Y-%m-%d')
#         #date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d'),range((end_date_dt.date() - start_date_dt.date()).days)) + [L[0][1][:10]] python3 不支持
#         date_range = [(start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range((end_date_dt.date() - start_date_dt.date()).days)] + [end[:10]]
#     else:
#         start_date_dt = start.date() if isinstance(start, datetime.datetime) else start
#         end_date_dt = end.date() if isinstance(end, datetime.datetime) else end
#         #date_range = map(lambda x: (start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d'), range((end_date_dt.date() - start_date_dt.date()).days)) + [L[0][1].date()]
#         date_range = [(start_date_dt + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in range((end_date_dt - start_date_dt).days)] + [end_date_dt.strftime('%Y-%m-%d')]
#     return date_range
# filter_sp = filter_sp.withColumn('dates', F.udf(getSparkDateRangeP2, ArrayType(StringType()))(F.col('start_date'),F.col('end_date')))
# filter_sp = filter_sp.withColumn('date', F.explode(F.col('dates')))
# filter_sp = filter_sp.drop('dates').drop('start_date').drop('end_date')

# com_data.fillna(0, subset=['total', 'real'])

# def groupSpark(sp, group_by, in_cols, out_cols, func=None, sep='|', kwargs=None):
#     """
#     按照 sp 的 group_by 传入函数进行计算
#     :param sp: spark dataframe
#     :param group_by: group by 列
#     :param in_cols: 传入 group by 计算的列
#     :param out_cols: 传出结果的列，除了 group_by 以外
#     :param func: 需要 group by 计算的函数
#     :param sep: 分割符
#     :param kwargs: func 的其他参数
#     :return: spark dataframe
#     """
#     key_len = len(group_by)
#     res_sp = sp.select(group_by + in_cols).rdd.map(lambda x: (sep.join(list(x)[:key_len]), [list(x)[key_len:]])).reduceByKey(lambda x,y: x+y). \
#         mapValues(lambda x: func(x, **kwargs)). \
#         flatMapValues(lambda x:x).map(lambda x: x[0].split(sep)+x[1]). \
#         toDF(group_by + out_cols)
#     return res_sp


# def test2(L, cols_name):
#     # raise Exception('~~~~~~~~~~~~{0}\n{1}\n'.format(L, cols_name))
#     org_data = map(list, L)
#     cols_name = eval(cols_name) if isinstance(cols_name, (str, unicode)) else cols_name
#     df = pd.DataFrame(org_data, columns=cols_name)
#     order_list = cols_name
#     print(df)
#     return [[1, 1]]
# bundle_date_sp.groupBy(['item_sku_id', 'Date']).agg(F.udf(test2)(F.collect_list(F.struct(F.col('amount_xxx_xxx'), F.col('amount_xxx_sgp'))), F.lit(str(t1_cols)))).show()


# for i, each_col in enumerate(out_cols_name):
#     max_prom_sp = max_prom_sp.join(t2, (max_prom_sp['item_sku_id'] == t2['item_sku_id']) & (max_prom_sp['Date'] == t2['Date']) &
#                                    (max_prom_sp[each_col] == t2['promotion_id']), 'left'). \
#         drop(t2['item_sku_id']).drop(t2['Date']).drop(t2['promotion_id']). \
#         withColumnRenamed('day_weight', prom_prefix_list[i]+'_day_weight'). \
#         withColumnRenamed('day_begin', prom_prefix_list[i]+'_day_begin')

