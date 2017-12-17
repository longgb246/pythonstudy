#-*- coding:utf-8 -*-
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
import os


app_name = 'sales_std'
conf = SparkConf().setAppName(app_name).setMaster('yarn')
try:
    sc = SparkContext(conf=conf).getOrCreate()
    sqlContext = SQLContext(sc)
except:
    pass
spark = SparkSession.builder.master('yarn').appName(app_name).enableHiveSupport().getOrCreate()


## hive sql
sql_1 = '''select * from app.app_ipc_ioa_sales_rdc where dt='2016-04-01' '''
sales_rdc = spark.sql(sql_1)

## DataFrame Operate
sales_rdc.filter(col('dc_id')=='772').show()                                                # filter by a Column
sales_rdc.filter((col('dc_id')=='772') & (col('item_first_cate_cd')=='1620')).show()        # filter by some Columns
sales_rdc.filter((col('dc_id')=='772') & (col('item_first_cate_cd')=='1620') & (col('total_sales')!=0)).show()

data = [[2,3,4], [1,2,3], [7,6,5]]
data_df = spark.createDataFrame(data, list('abc'))                                          # create a DF, with columns name
data_df2 = spark.createDataFrame(data)                                                      # create a DF


data = [[2,3,4], [1,2,3], [7,6,5]]


sqlContext.registerDataFrameAsTable(data_df2, "test_table")                                 # register a Tmp Table
test_data = spark.sql('select * from test_table')
# sqlContext.dropTempTable("test_table")

sqlContext.udf.register("stringLengthInt", lambda x: len(str(x)), IntegerType())            # register a Function for SQL
sqlContext.registerFunction("stringLengthInt", lambda x: len(str(x)), IntegerType())
sqlContext.sql("SELECT stringLengthInt('test') as len").show()
sqlContext.sql("SELECT stringLengthInt(a) as len from test_table ").show()

df_as1 = data_df.alias("df_as1")                                                            # alias
df_as2 = data_df.alias("df_as2")
joined_df = df_as1.join(df_as2, col("df_as1.a") == col("df_as2.a"), 'inner')                # 保留了全部列名
joined_df.select("df_as1.a", "df_as2.a", "df_as2.b", "df_as2.c").show()

print(data_df.columns)

# ---------------------------------------------------------------------------------
data1 = [[2, u'Alice'], [5, u'Bob']]
data2 = [[u'Tom', 80], [u'Bob', 85]]
data3 = [[2, 2, u'Alice'], [5, 5, u'Bob'], [5, 53, u'Bob'], [7, 1, u'Alice']]
data4 = [[2, 2, u'Alice'], [5, None, u'Bob'], [5, 53, None], [7, 1, u'Alice']]
df1 = spark.createDataFrame(data1, ["age", "name"])
df2 = spark.createDataFrame(data2, ["name", "height"])
df3 = spark.createDataFrame(data3, ['weight', 'height', 'name'])
df4 = spark.createDataFrame(data4, ['age', 'height', 'name'])
# ---------------------------------------------------------------------------------

df1.crossJoin(df2.select("height")).select("age", "name", "height").show()                  # crossJoin

df1.describe().show()                                                                       # Descriptive statistical analysis
df1.distinct().show()                                                                       # Distinct 无参数
df1.join(df2, 'name', 'inner').drop('age', 'height').show()                                 # Drop Columns

df3.dropDuplicates(['name', 'height']).show()                                               # dropDuplicates  drop_duplicates
df3.select('name').drop_duplicates().show()

df3.dropna().show()                                                                         # dropna 'any' 'all'
df3.na.drop().show()

print(df3.dtypes)

df4.fillna({'height': 50, 'name': 'unknown'}).show()                                        # fillna  dict value
df4.na.fill(50).show()
df4.groupby(['name', df4.age]).agg({'age': 'mean', 'name':'count'}).show()                  # groupBy  groupby

df3.join(df4, df3.name == df4.name, 'outer').select(df3.name, df3.height).show()            # join

# 存储文件信息，在 Spark 中，存储于 HDFS 上。不建议使用 saveAsTextFile，因为无法覆盖overwrite。
sql_3 = '''select
        item_first_cate_cd,
        item_second_cate_cd,
        count(item_second_cate_cd) as cate2_count
from
        gdm.gdm_m03_item_sku_da
where
        dt = '2017-11-14'
        and sku_valid_flag = 1
        and sku_status_cd = '3001'
        and data_type = 1
group by
        item_first_cate_cd,
        item_second_cate_cd
'''
data4 = spark.sql(sql_3)
data4.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(r'hdfs://ns15/user/cmo_ipc/longguangbin/work/count_cates/cate2' + os.sep + 'cate2')
sql_3 = '''select
        item_first_cate_cd,
        count(item_first_cate_cd) as cate2_count
from
        gdm.gdm_m03_item_sku_da
where
        dt = '2017-11-14'
        and sku_valid_flag = 1
        and sku_status_cd = '3001'
        and data_type = 1
group by
        item_first_cate_cd
'''
data4 = spark.sql(sql_3)
data4.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(r'hdfs://ns15/user/cmo_ipc/longguangbin/work/count_cates/cate1' + os.sep + 'cate1')
data2 = data1.limit(1000)
data2.write.csv(r'hdfs://ns15/user/cmo_ipc/longguangbin/work/count_cates/data_csv', header=True, mode="overwrite")         # 存储为 csv
data3 = spark.read.csv(r'hdfs://ns15/user/cmo_ipc/longguangbin/work/count_cates/data_csv')


# ---------------------------------------------------------
# 172.21.9.69:50010
# 172.21.9.226:50010

# 172.21.17.99:50010
# 172.21.17.102:50010

# 172.21.18.166:50010
# 172.21.18.168:50010

# 172.21.26.165:50010

# 172.21.29.3:50010
# 172.21.29.227:50010
# 172.21.29.235:50010

# 172.21.78.194:50010
# 172.21.78.199:50010
# ---------------------------------------------------------


# ---------------------------------------------------------
# 科学算法集市
import numpy as np
import pandas as pd

df1.show()
df1.select(df1.columns).show()
np.array(df1.select(df1.columns).collect())

df1.select(df1.columns).toArray()
mm = np.matrix(df1.toPandas())


from py_offline_ipc_ioa_inv_loc_cost_cal import CostCal, getSampleData
sku, Q, P_order, P_store, P_trans, Tau, v, weight, p_sale, beta, Const_num, Const_p, A_Matrix, T_period, r_ratio = getSampleData()


A_Matrix = spark.createDataFrame(A_Matrix)
Q = spark.createDataFrame(Q)
P_store = spark.createDataFrame(map(lambda x: [x], P_store))
P_order = spark.createDataFrame(map(lambda x: [x], P_order))
P_trans = spark.createDataFrame(P_trans)
Tau = spark.createDataFrame(Tau)
Const_num = spark.createDataFrame(map(lambda x: [x], Const_num))
Const_p = spark.createDataFrame(map(lambda x: [x], Const_p))


def calRes(sku, Q, P_order, P_store, P_trans, Tau, v, weight, p_sale, beta, Const_num, Const_p, T_period, r_ratio):
    cost_cal = CostCal(sku, Q, P_order, P_store, P_trans, Tau, v, weight, p_sale, beta, Const_num, Const_p, T_period, r_ratio)
    total_cost, total_cost_list = cost_cal.run(A_Matrix)
    return total_cost


A_Matrix = A_Matrix.toPandas()
Q = Q.toPandas()


P_store = P_store.toPandas()        # store     [ broadcast ]
P_order = P_order.toPandas()        # order     [ broadcast ]
P_trans = P_trans.toPandas()        # trans     [ broadcast ]
Tau = Tau.toPandas()                # tau       [ broadcast ]
Const_num = Const_num.toPandas()    # no need   []
Const_p = Const_p.toPandas()        # no need   []


calRes(sku, Q, P_order, P_store, P_trans, Tau, v, weight, p_sale, beta, Const_num, Const_p, T_period, r_ratio)


P_store_broadcast = sc.broadcast(P_store)
P_order_broadcast = sc.broadcast(P_order)
P_trans_broadcast = sc.broadcast(P_trans)
Tau_broadcast = sc.broadcast(Tau)
Const_num_broadcast = sc.broadcast(Const_num)
Const_p_broadcast = sc.broadcast(Const_p)




