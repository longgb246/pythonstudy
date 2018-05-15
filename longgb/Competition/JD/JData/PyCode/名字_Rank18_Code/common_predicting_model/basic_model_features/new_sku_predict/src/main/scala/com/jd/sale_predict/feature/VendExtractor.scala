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


object VendExtractor extends Serializable{
  def extract(sc: SparkContext, projectPath: String) = {
    // kucun
    val vendPath = projectPath + "/data/{train_sku_vendibility_daily.csv,sku_vendibility_daily.csv}"
    val skuVendData = GBKtoUtf8.transfer(sc, vendPath).repartition(100).distinct.map(
      _.split(",")
    ).filter( x =>
      (x(4).trim.toInt == 1 && (
        x(2).trim.toDouble > 0 || x(3).trim.toInt == 1)
      )
    ).map { x =>
      val stat_date = x(0)
      val dim_delv = x(1)
      val stock_qtty = x(2).trim.toDouble
      val allow_reserve_flg = x(3).trim.toInt
      val sku_statuc_cd = x(4).trim.toInt
      val dt = x(5)
      val sku_id = x(6).trim
      (sku_id, stat_date)
    }.distinct.groupByKey().filter { x =>
      x._2.toList.sortWith(_ < _)(0) > "2016-02-28"
    }
    skuVendData
  }

  def extractFirstDt(skuVendData: RDD[(String, Iterable[String])]) = {
    val skuFirstDt = skuVendData.map { x =>
      val vList = x._2.toList.sortWith(_ < _ )
      (x._1, vList(0))
    }
    skuFirstDt
  }

}
