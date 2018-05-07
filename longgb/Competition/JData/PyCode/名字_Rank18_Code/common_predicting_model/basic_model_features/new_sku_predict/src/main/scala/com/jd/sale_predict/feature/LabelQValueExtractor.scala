package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar
import scala.math._


object LabelQValueExtractor extends Serializable{

  def extract(skuLabel: RDD[(String, (String, Double))],
    infoData: RDD[(String, Info)], keyType: String) = {
    val sameData = skuLabel.join(infoData).map { x =>
      val sku = x._1
      val info = x._2._2
      val keyString = SaleQValueExtractor.extractKeyString(keyType, info)
      val firstDt = x._2._1._1
      val sale = x._2._1._2
      ((keyString, sku, firstDt), sale)
    }.groupByKey().map(x => (x._1._1,(x._1._2, x._1._3, x._2.toList.max))).groupByKey().flatMap { x =>
      val v = x._2.toList.sortWith(_._2 < _._2)
      var allSale = 0.0
      var count = 0
      var result = new ListBuffer[(String, List[Double])]
      v.foreach { i =>
        val avgSale = if(count > 0) (allSale / count) else 0
        result += ((i._1, List(avgSale)))
        if (i._3 > 0.0) {
          allSale += i._3
          count += 1
        }
      }
      result.toList
    }
    sameData
  }
}
