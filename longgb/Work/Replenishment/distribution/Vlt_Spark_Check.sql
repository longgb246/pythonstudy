-- 结果表
select 
	*
from
	dev.lgb_neipeiOrder_2;
[select * from dev.lgb_neipeiOrder_2 limit 10;]
[select * from dev.lgb_vlt_jobCombinedData_2 limit 10;]
[select * from dev.lgb_sku_result_2 limit 10;]
[select * from dev.lgb_cid3_result_2 limit 10;]
[select * from dev.lgb_result_2 limit 10;]


-- lgb_neipeiOrder_2  对比  lgb_tmp_neipeiOrder_52
-- lgb_neipeiOrder_3  对比  lgb_tmp_neipeiOrder_53
select
	count(pur_bill_id)
from
	dev.lgb_neipeiOrder_2;
[ select count(pur_bill_id) as count_num from dev.lgb_neipeiOrder_3; ] # 2:7529  3:7568
[ select count(pur_bill_id) as count_num from dev.lgb_tmp_neipeiOrder_53; ] # 2:7529  3:7568

select * from dev.lgb_tmp_orders_52_2 limit 10;


-- 导入
from pyspark import SparkContext,SparkConf,HiveContext
from datetime import datetime as dat,timedelta
import pyspark.sql.functions as func
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import os
appName = "VLT_OPTIMIZATION"
mode = "yarn-client"
conf = SparkConf().setAppName(appName).setMaster("yarn-client")
sc = SparkContext(conf=conf)
hiveContext = HiveContext(sc)
-- 删除表
hiveContext.sql("drop table if EXISTS dev.lgb_tmp_sourceOrder_52_2")
lgb_tmp_sourceOrder_52_2 = hiveContext.sql("select  item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_sourceOrder_52 a left join dev.lgb_tmp_neipeiOrder_52 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null").coalesce(10) ]
lgb_tmp_sourceOrder_52_2.write.mode("overwrite").saveAsTable("dev.lgb_tmp_sourceOrder_52_2")

aa_name = hiveContext.sql("select pur_bill_id, sku_id, into_wh_tm from gdm.gdm_m04_pur_recv_det_basic_sum where dt='2017-05-01' AND cgdetail_yn=1 AND into_wh_qtty>0").coalesce(10)
aa_name.write.mode("overwrite").saveAsTable("dev.lgb_tmp_into_wareHouse_52")






-- New check
into_wareHouse = hiveContext.table("gdm.gdm_m04_pur_recv_det_basic_sum").where("dt='2017-05-01'" + " AND cgdetail_yn=1 AND into_wh_qtty>0 ").groupBy("pur_bill_id", "sku_id").agg(func.min("into_wh_tm").alias("into_wh_tm")).coalesce(10)
orders = hiveContext.sql("select * from lgb_tmp_orders_52_2").coalesce(10)
into_wareHouse = orders.join(into_wareHouse,["pur_bill_id","sku_id"]).coalesce(10)
t_6_Old = hiveContext.table("fdm.fdm_procurement_po_process").where("po_yn=1 AND po_state=6 AND process_desc LIKE '%采购单提交成功，启动审核工作流%' ").groupBy("po_id").agg(func.max("create_time").alias("t_6")).coalesce(10)
t_6_New = hiveContext.table("fdm.fdm_procurement_lifecycle_chain").where("dt='4712-12-31' and actiontype=104 and yn=1 ").groupBy("poid").agg(func.max("createtime").alias("t_6")).coalesce(10)
t_6 = t_6_Old.unionAll(t_6_New)





