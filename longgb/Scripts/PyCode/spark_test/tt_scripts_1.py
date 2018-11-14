# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/11/14
  Usage   :
"""


def get_seasonal_flag(*x):
    return [1]


def test_a():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    spark = SparkSession.builder.getOrCreate()

    df1 = spark.createDataFrame([[1, 4, 1, 3],
                                 [1, 4, 2, 4],
                                 [1, 4, 3, 3],
                                 [1, 4, 4, 34],
                                 [1, 4, 5, 35],
                                 [1, 4, 6, 23],
                                 [1, 4, 7, 34],
                                 [1, 4, 8, 37],
                                 [1, 4, 9, 53],
                                 [1, 4, 10, 63],
                                 [1, 4, 11, 23],
                                 [1, 4, 12, 73],
                                 [2, 5, 1, 23],
                                 [2, 5, 2, 24],
                                 [2, 5, 3, 23],
                                 [2, 5, 4, 324],
                                 [2, 5, 5, 325],
                                 [2, 5, 6, 223],
                                 [2, 5, 7, 324],
                                 [2, 5, 8, 327],
                                 [2, 5, 9, 523],
                                 [2, 5, 10, 623],
                                 [2, 5, 11, 223],
                                 [2, 5, 12, 723],
                                 ],
                                ['sku', 'store', 'month', 'sales'])
    df1.show()

    # df1.groupBy(['sku']). \
    #     agg(F.udf(get_seasonal_flag)(F.collect_list(F.struct(F.col('month'), F.col('sales'))), F.lit('2018-05-03'))). \
    #     rdd.flatMap(lambda x: _deal_ts(list(x), scc_dict, adjust_circle))

    from pyspark.sql.types import ArrayType, StringType
    # ArrayType(ArrayType())

    # df2 = df1.groupBy(['sku', 'store']). \
    #     agg(F.udf(get_seasonal_flag, ArrayType(StringType()))
    #         (F.collect_list(F.struct(F.col('month'), F.col('sales'))), F.lit('2018-05-03')).
    #         alias('data'))

    # df2.show()
    # df2.withColumn('_tmp_data', F.explode(F.col('data'))).drop('data').show()

    # label_data = unlabel_data.groupBy(self.key).agg(F.collect_list(F.col(self.ds)).alias(self.ds)).rdd.flatMap(
    #     lambda x: _deal_ts(list(x), scc_dict, adjust_circle))

    key = ['sku', 'store']
    key = ['sku']

    df1.groupBy(key).agg(F.collect_list(F.struct(F.col('month'), F.col('sales')))). \
        rdd.flatMap(lambda x: get_seasonal_flag(list(x), '2018-05-03')). \
        toDF(key + ['month', 'flag', 'sales']).show(40)

    df1.groupBy(key).agg(F.collect_list(F.struct(F.col('month'), F.col('sales')))). \
        rdd.flatMap(lambda x: get_seasonal_flag(list(x))). \
        toDF(key + ['month', 'flag', 'sales']).show(40)

    pass
