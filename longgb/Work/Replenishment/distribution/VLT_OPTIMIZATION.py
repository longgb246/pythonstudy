#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'gaoyun3'
from pyspark import SparkContext,SparkConf,HiveContext
from datetime import datetime as dat,timedelta
import pyspark.sql.functions as func
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import os
# import logging

appName = "VLT_OPTIMIZATION"
mode = "yarn-client"
conf = SparkConf().setAppName(appName).setMaster("yarn-client")
sc = SparkContext(conf=conf)
hiveContext = HiveContext(sc)
# logging.basicConfig(filename="/home/cmo_ipc/VLT_OPTIMIZATION/log/"+appName+".log",level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s")

class VLT_Distribution:
    def __init__(self, this_year=dat.today().year, this_month=dat.today().month, this_day=dat.today().day):
        self.setTime(this_year, this_month, this_day)
        print self.dt
    def setTime(self, this_year, this_month, this_day):
        self.this_year = this_year
        self.this_month = this_month
        self.this_day = this_day
        self.dt = dat.strftime(dat(this_year, this_month, this_day).date() + timedelta(days=-1),format="%Y-%m-%d")
        self.dtStr = "'"+self.dt+"'"
        self.beginYearDate = "'"+dat(this_year-1, this_month, this_day).date().strftime("%Y-%m-%d")+"'"
        self.orderCount = 10
    def log(self,dt,types,msg,currTime=dat.now().strftime("%Y-%m-%d %H:%M:%S")):
        def log2Hive():
            log =hiveContext.createDataFrame([{"dt":dt,"types":types,"message":msg,"currtime":currTime}]) #types: "INFO" ,"ERROR"
            log.write.mode("append").insertInto("app.app_vlt_distribution_log")
        log2Hive()
    def run(self):
        print 'extractPurChasedOrder'
        self.extractPurChasedOrder()
        print 'removeNeipei'
        self.removeNeipei()
        print 'sourcePo'
        self.sourcePo =self.sourceOrder.select("pur_bill_id").distinct().coalesce(10)
        print 'extract_into_wareHouse'
        self.extract_into_wareHouse()
        print 'extract_t_6'
        self.extract_t_6()
        print 'extract_combine_Data'
        self.extract_combine_Data()
        print 'transform_data'
        self.transform_data()
        print 'save'
        self.save()
    def extractPurChasedOrder(self):
        try:
            self.sourceOrder =   hiveContext.table("gdm.gdm_m04_pur_det_basic_sum").where("dt="+self.dtStr + " and valid_flag='1' AND cgdetail_yn=1 AND create_tm BETWEEN "+self.beginYearDate+" AND "+self.dtStr+" AND complete_dt <>'' AND pur_bill_src_cd IN (2, 3, 4, 10)").select("item_third_cate_cd","pur_bill_id" ,"sku_id" ,"supp_brevity_cd" ,"int_org_num" ,"create_tm" ,"complete_dt").coalesce(10)
            self.autoPoOrder =   hiveContext.table("gdm.gdm_m04_pur_det_basic_sum").where("dt="+self.dtStr + " and valid_flag='1' AND cgdetail_yn=1 AND create_tm BETWEEN "+self.beginYearDate+" AND "+self.dtStr+" AND complete_dt <>'' AND pur_bill_src_cd =15").select("item_third_cate_cd","pur_bill_id" ,"sku_id" ,"supp_brevity_cd" ,"int_org_num" ,"create_tm" ,"complete_dt").coalesce(10)
            self.orders = self.sourceOrder.unionAll(self.autoPoOrder) #42,633,086
        except Exception as e:
            raise Exception(e.message)
    def removeNeipei(self):
        try:
            unqiueOrder = self.orders.select("pur_bill_id").distinct()
            fdm_scm_cgfenpei_chain = unqiueOrder.join(hiveContext.table("fdm.fdm_scm_cgfenpei_chain").select(col("rfid").alias("pur_bill_id"),"idcompany"),["pur_bill_id"]).coalesce(10)
            fdm_scm_cgtable_chain = unqiueOrder.join(hiveContext.table("fdm.fdm_scm_cgtable_chain").select(col("id").alias("pur_bill_id"),"idcompany"),["pur_bill_id"]).coalesce(10)
            neipeiOrder=fdm_scm_cgfenpei_chain.join(fdm_scm_cgtable_chain,(fdm_scm_cgfenpei_chain.pur_bill_id==fdm_scm_cgtable_chain.pur_bill_id) & (fdm_scm_cgfenpei_chain.idcompany<>fdm_scm_cgtable_chain.idcompany)).select(fdm_scm_cgfenpei_chain.pur_bill_id).distinct().coalesce(10)
            temp_neipeiOrder = "lgb_neipeiOrder_"+str(self.this_day)
            hiveContext.sql("drop table if exists dev."+temp_neipeiOrder)
            os.system("hadoop fs -rm -r -skipTrash dev.db/" + str.lower(temp_neipeiOrder))
            neipeiOrder.write.saveAsTable("dev."+temp_neipeiOrder)
            neipeiOrder=hiveContext.table("dev."+temp_neipeiOrder)
            self.sourceOrder = self.sourceOrder.join(neipeiOrder,neipeiOrder.pur_bill_id==self.sourceOrder.pur_bill_id,how="left_Outer").where(neipeiOrder.pur_bill_id.isNull()).select(col("item_third_cate_cd"),self.sourceOrder.pur_bill_id,col("sku_id"),col("supp_brevity_cd"),col("int_org_num"),col("create_tm"),col("complete_dt")).coalesce(10)
            self.autoPoOrder = self.autoPoOrder.join(neipeiOrder,neipeiOrder.pur_bill_id==self.autoPoOrder.pur_bill_id,how="left_Outer").where(neipeiOrder.pur_bill_id.isNull()).select(col("item_third_cate_cd"),self.autoPoOrder.pur_bill_id,col("sku_id"),col("supp_brevity_cd"),col("int_org_num"),col("create_tm"),col("complete_dt")).coalesce(10)
            self.orders = self.orders.join(neipeiOrder,neipeiOrder.pur_bill_id==self.orders.pur_bill_id,how="left_Outer").where(neipeiOrder.pur_bill_id.isNull()).select(col("item_third_cate_cd"),self.orders.pur_bill_id,col("sku_id"),col("supp_brevity_cd"),col("int_org_num"),col("create_tm"),col("complete_dt")).coalesce(10)
        except Exception as e:
            raise Exception(e.message)
    def extract_into_wareHouse(self):
        try:
            into_wareHouse = hiveContext.table("gdm.gdm_m04_pur_recv_det_basic_sum").where("dt="+self.dtStr + " AND cgdetail_yn=1 AND into_wh_qtty>0 ").groupBy("pur_bill_id", "sku_id").agg(func.min("into_wh_tm").alias("into_wh_tm")).coalesce(10)
            self.into_wareHouse = self.orders.join(into_wareHouse,["pur_bill_id","sku_id"]).coalesce(10)
        except Exception as e:
            raise Exception(e.message)
    def extract_t_6(self):
        try:
            t_6_Old = hiveContext.table("fdm.fdm_procurement_po_process").where("po_yn=1 AND po_state=6 AND process_desc LIKE '%采购单提交成功，启动审核工作流%' ").groupBy("po_id").agg(func.max("create_time").alias("t_6")).coalesce(10)
            t_6_New = hiveContext.table("fdm.fdm_procurement_lifecycle_chain").where("dt='4712-12-31' and actiontype=104 and yn=1 ").groupBy("poid").agg(func.max("createtime").alias("t_6")).coalesce(10)
            t_6 = t_6_Old.unionAll(t_6_New)
            t_6_source = self.sourcePo.join(t_6,t_6.po_id==self.sourcePo.pur_bill_id).select("pur_bill_id","t_6").coalesce(10)
            t_6_autoPo = self.autoPoOrder.groupBy("pur_bill_id").agg(func.max("create_tm").alias("t_6")).coalesce(10)
            self.t6 = t_6_source.unionAll(t_6_autoPo).coalesce(10).groupBy("pur_bill_id").agg(func.max("t_6").alias("t_6")).coalesce(10)
        except Exception as e:
            raise Exception(e.message)
    def extract_combine_Data(self):
        try:
            combinedData=self.into_wareHouse.join(self.t6,["pur_bill_id"]).coalesce(10)
            temp_vlt_jobCombinedData= "lgb_vlt_jobCombinedData_"+str(self.this_day)
            hiveContext.sql("drop table if exists dev."+temp_vlt_jobCombinedData)
            os.system("hadoop fs -rm -r -skipTrash dev.db/" + str.lower(temp_vlt_jobCombinedData))
            combinedData.write.saveAsTable("dev."+temp_vlt_jobCombinedData)
            self.combinedData = hiveContext.table("dev."+temp_vlt_jobCombinedData).select("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","into_wh_tm","t_6",round((unix_timestamp("into_wh_tm")-unix_timestamp("t_6"))/86400.0,2).alias("vlt")).where(col("vlt").between(0.5,60)).select("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","into_wh_tm","t_6",round("vlt").alias("vlt")).coalesce(10)
        except Exception as e:
            raise Exception(e.message)
    def transform_data(self):
        try:
            sku_slice_vlt_count = self.combinedData.groupBy("sku_id","supp_brevity_cd","int_org_num","vlt") .agg(func.countDistinct("pur_bill_id","sku_id").alias("sku_vlt_OrderCount")).coalesce(10)
            sku_slice_count     = self.combinedData.groupBy("sku_id","supp_brevity_cd","int_org_num")       .agg(func.max("item_third_cate_cd").alias("item_third_cate_cd"),func.countDistinct("pur_bill_id","sku_id").alias("sku_Ordercount"),round(func.mean("vlt"),2).alias("sku_vlt_mean"),round(func.stddev("vlt"),2).alias("sku_vlt_stdev")).coalesce(10)
            skuData = sku_slice_vlt_count.join(sku_slice_count,["sku_id","supp_brevity_cd","int_org_num"]).select("sku_id","supp_brevity_cd","int_org_num","item_third_cate_cd","vlt","sku_vlt_OrderCount","sku_Ordercount",round(col("sku_vlt_OrderCount")*1.0/col("sku_Ordercount"),2).alias("sku_vlt_prob"),"sku_vlt_mean","sku_vlt_stdev").coalesce(10)
            sku_result = skuData.groupBy("sku_id","supp_brevity_cd","int_org_num").agg(func.max("item_third_cate_cd").alias("item_third_cate_cd"),func.collect_set(func.concat(col("vlt").cast("string"),lit(":"),col("sku_vlt_prob").cast("string"))).alias("sku_vlt_dist"),func.max("sku_Ordercount").alias("sku_Ordercount"),func.max("sku_vlt_mean").alias("sku_vlt_mean"),func.max("sku_vlt_stdev").alias("sku_vlt_stdev")).coalesce(10)
            order_lt_10         = sku_slice_count.where(col("sku_Ordercount")< self.orderCount).select("supp_brevity_cd","int_org_num","item_third_cate_cd").distinct().coalesce(10)
            temp_sku_result = "lgb_sku_result_"+str(self.this_day)
            hiveContext.sql("drop table if exists dev."+temp_sku_result)
            os.system("hadoop fs -rm -r -skipTrash dev.db/" + str.lower(temp_sku_result))
            sku_result.write.saveAsTable("dev."+temp_sku_result)
            sku_result = hiveContext.table("dev."+temp_sku_result)
            cid3_slice_vlt_count = self.combinedData.groupBy("item_third_cate_cd","supp_brevity_cd","int_org_num","vlt").agg(func.countDistinct("pur_bill_id","sku_id").alias("cid3_vlt_Count")).coalesce(10)
            cid3_slice_count     = self.combinedData.groupBy("item_third_cate_cd","supp_brevity_cd","int_org_num") .agg(func.countDistinct("pur_bill_id","sku_id").alias("cid3_Ordercount"),round(func.mean("vlt"),2).alias("cid3_vlt_mean"),round(func.stddev("vlt"),2).alias("cid3_vlt_stdev")).coalesce(10)
            cid3_data = cid3_slice_vlt_count.join(cid3_slice_count,["item_third_cate_cd","supp_brevity_cd","int_org_num"]).select("item_third_cate_cd","supp_brevity_cd","int_org_num","vlt","cid3_vlt_Count","cid3_Ordercount",round(col("cid3_vlt_Count")*1.0/col("cid3_Ordercount"),2).alias("cid3_vlt_prob"),"cid3_vlt_mean","cid3_vlt_stdev").coalesce(10)
            cid3_result = order_lt_10.join(cid3_data,["item_third_cate_cd","supp_brevity_cd","int_org_num"]).groupBy("item_third_cate_cd","supp_brevity_cd","int_org_num").agg(func.collect_set(func.concat(col("vlt").cast("string"),lit(":"),col("cid3_vlt_prob").cast("string"))).alias("cid3_vlt_dist"),func.max("cid3_Ordercount").alias("cid3_Ordercount"),func.max("cid3_vlt_mean").alias("cid3_vlt_mean"),func.max("cid3_vlt_stdev").alias("cid3_vlt_stdev")).coalesce(10)
            temp_cid3_result="lgb_cid3_result_"+str(self.this_day)
            hiveContext.sql("drop table if exists dev."+temp_cid3_result)
            os.system("hadoop fs -rm -r -skipTrash dev.db/" + str.lower(temp_cid3_result))
            cid3_result.write.saveAsTable("dev."+temp_cid3_result)
            cid3_result = hiveContext.table("dev."+temp_cid3_result)
            self.result=sku_result.join(cid3_result,["item_third_cate_cd","supp_brevity_cd","int_org_num"],how="left_outer").select(
                "item_third_cate_cd",
                "supp_brevity_cd",
                "int_org_num",
                "sku_id",
                "sku_vlt_dist",
                "sku_Ordercount",
                "sku_vlt_mean",
                "sku_vlt_stdev",
                F.when(sku_result.sku_Ordercount<10,cid3_result.cid3_vlt_dist).otherwise(sku_result.sku_vlt_dist).alias("dist"),
                F.when(sku_result.sku_Ordercount<10,cid3_result.cid3_Ordercount).otherwise(sku_result.sku_Ordercount).alias("Ordercount"),
                F.when(sku_result.sku_Ordercount<10,cid3_result.cid3_vlt_mean).otherwise(sku_result.sku_vlt_mean).alias("vlt_mean"),
                F.when(sku_result.sku_Ordercount<10,cid3_result.cid3_vlt_stdev).otherwise(sku_result.sku_vlt_stdev).alias("vlt_stdev"),
            ).coalesce(10)
            self.result=self.result.withColumn("dt",lit(self.dt))
        except Exception as e:
            raise Exception(e.message)
    def save(self):
        try:
            temp_result = "lgb_result_"+str(self.this_day)
            hiveContext.sql("drop table if exists dev." + temp_result)
            os.system("hadoop fs -rm -r -skipTrash dev.db/" + str.lower(temp_result))
            self.result.write.saveAsTable("dev." + temp_result)
        except Exception as e:
            raise Exception(e.message)

if __name__ == "__main__":
    dist = VLT_Distribution(2017, 5, 2)
    dist.run()


