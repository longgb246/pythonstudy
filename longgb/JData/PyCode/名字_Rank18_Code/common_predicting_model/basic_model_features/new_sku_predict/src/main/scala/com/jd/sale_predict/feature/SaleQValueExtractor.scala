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

object SaleQValueExtractor extends Serializable{
  def extractKeyString(keyType: String, info: Info) = {
    val keyString = keyType match {
      case "cid1" => info.cid1
      case "cid2" => info.cid2
      case "cid3" => info.cid3
      case "brand" => info.brand
      case "itemType" => info.itemType
      case "spu" => info.spu
      case "brand-cid1" => List(info.brand, info.cid1).mkString("-")
      case "brand-cid2" => List(info.brand, info.cid2).mkString("-")
      case "brand-cid3" => List(info.brand, info.cid3).mkString("-")
      case "brand-itemType" => List(info.brand, info.itemType).mkString("-")
      case "brand-cid3-iT" => List(info.brand, info.cid3, info.itemType).mkString("-")
      case "cid3-iT" => List(info.cid3, info.itemType).mkString("-")
    }
    keyString
  }
  def extractKeyDt(skuFirstDt: RDD[(String, String)], infoData: RDD[(String, Info)],
    keyType: String, dtNum: Int) = {
    val keyDt = skuFirstDt.join(infoData).flatMap { x =>
      val sku = x._1
      val firstDt = x._2._1
      val info = x._2._2
      val result = Range(dtNum * -1, 0).map { i =>
        val format = new SimpleDateFormat("yyyy-MM-dd")
        val firstDate = format.parse(firstDt)
        val dt = {
          val cal = Calendar.getInstance
          cal.setTime(firstDate)
          cal.add(Calendar.DATE, i)
          format.format(cal.getTime)
        }
        val keyString = extractKeyString(keyType, info)
        ((keyString, dt), sku)
      }
      result
    }.distinct
    keyDt
  } 
  def extract(skuFirstDt: RDD[(String, String)], infoData: RDD[(String, Info)],
    saleData: RDD[((String, String), Double)], keyType: String, dtNum: Int) = {
    val keyDt = extractKeyDt(skuFirstDt, infoData, keyType, dtNum)
    val keySale = saleData.map( x =>
      (x._1._1, (x._1._2, x._2))
    ).join(infoData).map { x =>
      val sku = x._1
      val info = x._2._2
      val dt = x._2._1._1
      val sale = x._2._1._2
      val keyString = extractKeyString(keyType, info)
      ((keyString, dt), (sale, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map { x =>
      (x._1, x._2._1 / x._2._2)
    }
    /*.groupByKey().flatMap { x =>
      val vList = x._2.toList.map(i => (i._1, log(i._2 + 1)))
      val saleList = vList.map(_._2)
      val minSale = saleList.min
      val maxSale = saleList.max
      val minusSale = maxSale - minSale
      vList.map { i =>
        val saleQ = if(minusSale > 0) ((i._2 - minSale) / minusSale) else 1.0
        ((i._1, x._1), saleQ)
      }
    }
    */
    val skuKeyQValue = keyDt.join(keySale).map { x =>
      val keyString = x._1._1
      val dt = x._1._2
      val sku = x._2._1
      val saleQ = x._2._2
      (sku, (saleQ, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map( x =>
      (x._1, List(x._2._1 / x._2._2))
    )
    skuKeyQValue
  }
}
