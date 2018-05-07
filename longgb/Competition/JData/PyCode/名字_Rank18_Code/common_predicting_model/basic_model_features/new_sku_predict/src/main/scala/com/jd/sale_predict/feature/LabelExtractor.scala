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


object LabelExtractor extends Serializable{

  def extractLabel(skuVendData: RDD[(String, Iterable[String])],
    saleData: RDD[((String, String), Double)]) = {
    val skuSevenDt = skuVendData.flatMap { x =>
      val vList = x._2.toList.sortWith(_ < _)
      val format = new SimpleDateFormat("yyyy-MM-dd")
      val firstDt = vList(0)
      val firstDate = format.parse(firstDt)
      val sevenDt = {
        val cal = Calendar.getInstance
        cal.setTime(firstDate)
        cal.add(Calendar.DATE, 7)
        format.format(cal.getTime)
      }
      val vfList = vList.filter(dt => (dt < sevenDt))
      val result = vfList.map(dt => ((x._1, dt), firstDt))
      result
    }

    val skuLabel = skuSevenDt.leftOuterJoin(saleData).map { x =>
      val sku = x._1._1
      val firstDt = x._2._1
      val sale = x._2._2.getOrElse(0.0)
      ((sku, firstDt), (sale, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map ( x =>
      (x._1._1, (x._1._2, x._2._1 / x._2._2))
    )
    skuLabel
  }
}
