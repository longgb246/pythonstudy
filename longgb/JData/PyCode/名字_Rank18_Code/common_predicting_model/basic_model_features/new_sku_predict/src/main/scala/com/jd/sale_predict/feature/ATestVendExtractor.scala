package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar


object ATestVendExtractor extends Serializable{
  def extractFirstDt(sc: SparkContext, projectPath: String) = {
    // kucun
    //val vendPath = projectPath + "/data/A_leaderBoard_onStock_Time.csv"
    val vendPath = projectPath + "/data/上柜日期.csv"
    val skuFirstDt = GBKtoUtf8.transfer(sc, vendPath).map(
      _.split(",")
    ).map { x =>
      val sku_id = x(0).trim
      val format1 = new SimpleDateFormat("yyyy/MM/dd")
      val format2 = new SimpleDateFormat("yyyy-MM-dd")
      val dt = format2.format(format1.parse(x(1)))
      (sku_id, dt)
    }.distinct
    skuFirstDt
  }

  def extract(skuFirstDt: RDD[(String, String)]) = {
    val skuVendData = skuFirstDt.flatMap { x =>
      val format = new SimpleDateFormat("yyyy-MM-dd")
      val firstDt = x._2
      val firstDate = format.parse(firstDt)
      Range(0, 7).map { i =>
        val dt = {
          val cal = Calendar.getInstance
          cal.setTime(firstDate)
          cal.add(Calendar.DATE, i)
          format.format(cal.getTime)
        }
        (x._1, dt)
      }
    }.groupByKey()
    skuVendData
  }

}
