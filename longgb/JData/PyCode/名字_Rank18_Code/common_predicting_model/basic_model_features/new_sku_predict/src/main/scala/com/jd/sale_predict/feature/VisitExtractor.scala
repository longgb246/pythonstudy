package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar


object VisitExtractor {
  def extract(sc: SparkContext, projectPath: String) = {
    //流量表
    val visitPath = projectPath + "/data/{train_sku_flow_city_daily.csv,sku_flow_city_daily.csv}"
    val visitData = GBKtoUtf8.transfer(sc, visitPath).distinct.map(
      _.split("\t")
    ).map { x =>
      val dt = x(0)
      val pv = if (x(7).trim != "") x(7).trim.toDouble else 0.0
      val uv = if (x(8).trim != "") x(8).trim.toDouble else 0.0
      val stay_time = if (x(9).trim != "") x(9).trim.toDouble else 0.0
      val sku_id = x(11).trim
      ((sku_id, dt), (pv, uv, stay_time))
    }.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))
    visitData
  }
}
