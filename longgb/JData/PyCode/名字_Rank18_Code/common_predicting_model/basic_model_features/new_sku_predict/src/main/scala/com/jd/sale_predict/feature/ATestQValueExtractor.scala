package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar
import scala.math._

object ATestQValueExtractor {
  def extractBrandCateDt(skuFirstDt: RDD[(String, String)],
    infoData: RDD[(String, Info)]) = {
    val brandCateDt = skuFirstDt.join(infoData).flatMap { x =>
      val sku = x._1
      val firstDt = "2017-02-28"
      val brandCate = x._2._2
      val result = Range(-7, 0).map { i =>
        val format = new SimpleDateFormat("yyyy-MM-dd")
        val firstDate = format.parse(firstDt)
        val dt = {
          val cal = Calendar.getInstance
          cal.setTime(firstDate)
          cal.add(Calendar.DATE, i)
          format.format(cal.getTime)
        }
        (sku, brandCate.brand, brandCate.cid3, dt)
      }
      result
    }
    brandCateDt
  }


}

