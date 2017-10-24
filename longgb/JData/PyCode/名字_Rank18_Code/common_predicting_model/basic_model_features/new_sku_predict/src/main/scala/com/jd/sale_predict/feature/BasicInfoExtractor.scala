package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar

case class Info(spu: String, brand: String, cid1: String, cid2: String,
  cid3: String, itemType: String, skuName: String, shelvesDt: String,colour: String)

object BasicInfoExtractor extends Serializable{
  def extract(sc: SparkContext, projectPath: String, fileName: String, delimiter: String) = {
    val infoPath = "%s/data/%s".format(projectPath, fileName)
    val infoData = GBKtoUtf8.transfer(sc, infoPath).distinct.map(
      _.split(delimiter)
    ).map { x => 
      val skuName = x(0)
      val itemType = x(29)
      val shelvesDt = x(21)
      val sku_id = x(42).trim
      val spu_id = x(43).trim
      val brandCode = x(4)
      val cid1 = x(14)
      val cid2 = x(16)
      val cid3 = x(18)
      val colour = x(38)
      (sku_id, Info(spu_id, brandCode, cid1, cid2, cid3, itemType, skuName, shelvesDt,colour))
    }.distinct
    infoData
  }
}

