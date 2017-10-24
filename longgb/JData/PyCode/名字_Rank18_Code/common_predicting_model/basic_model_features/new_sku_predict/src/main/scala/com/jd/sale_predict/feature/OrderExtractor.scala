package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar


object OrderExtractor extends Serializable{
  def extract(sc: SparkContext, projectPath: String) = {
    //订单表
    val ordPath = projectPath + "/data/{train_sku_sales_daily.csv,sku_sales_daily.csv}"
    val saleData = GBKtoUtf8.transfer(sc, ordPath).distinct.map(
      _.split(",")
    ).map { x =>
      val dt = x(0)
      val dc_id = x(1).trim.toInt
      val order_date = x(2)
      val total_sales = x(3).trim.toDouble
      val sku_id = x(4).trim
      ((sku_id, order_date), total_sales)
    }.reduceByKey(_+_)
    saleData
  }

}
