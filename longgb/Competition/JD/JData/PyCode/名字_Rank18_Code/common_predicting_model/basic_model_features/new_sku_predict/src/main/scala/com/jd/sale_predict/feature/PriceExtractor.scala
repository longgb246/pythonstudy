package com.jd.sale_predict.feature

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Calendar
import scala.math._

object PriceExtractor extends Serializable{
  def extract(sc: SparkContext, projectPath: String, outputPath: String,
    infoData: RDD[(String, Info)]) = {
    val pricePath = projectPath + "/data/{train_sku_price_daily.csv,sku_price_daily.csv}"
    val priceData = GBKtoUtf8.transfer(sc, pricePath).distinct.repartition(100).map(
      _.split(",")
    ).filter(_.length == 4).map { attributes => 
      val jd_prc = try {
        attributes(0).trim.toDouble
      } catch {
        case _: Throwable => -1.0
      }
      val mkt_prc = try {
        attributes(1).trim.toDouble
      } catch {
        case _: Throwable => -1.0
      }
      val dt = attributes(2)
      val sku_id = attributes(3).trim
      ((sku_id, jd_prc), dt)
    }.filter(_._1._2 > 0.0).groupByKey().map { x =>
      val vList = x._2.toList.sortWith(_ < _)
      (x._1._1, (x._1._2, vList(0)))
    }.groupByKey().map { x =>
      val vList = x._2.toList.sortWith(_._2 < _._2)
      (x._1, List(vList(0)._1))
    }
    /*
    val skuCatePrice = priceData.join(infoData).map { x =>
      val sku = x._1
      val price = x._2._1
      val thirdCate = x._2._2.cid3
      (sku, (thirdCate, price))
    }
    val catePrice = skuCatePrice.map(_._2).groupByKey().map { x =>
      val vList = x._2.toList
      val priceMin = vList.min
      val priceMax = vList.max
      (x._1, (priceMin, priceMax))
    }
    val catePriceString = catePrice.map(x => List(x._1, x._2._1, x._2._2).mkString("\t"))
    catePriceString.coalesce(1).saveAsTextFile(outputPath + "/feature/catePrice/")

    val catePriceCollect = catePrice.collectAsMap
    val catePriceBC = sc.broadcast(catePriceCollect)
    val skuPrice = skuCatePrice.mapPartitions { iters =>
      val catePriceMap = catePriceBC.value
      //val catePriceMap = Map("866" -> (0.01,9000.0), "655" -> (1.0,8.88888888E8), "1049" -> (0.19,104998.0), "672" -> (0.01,800000.0), "867" -> (2.0,4162.0))
      iters.map { x =>
        val cate = x._2._1
        val cp = catePriceMap.getOrElse(cate, (0.0, 1.0))
        val pMin = log(cp._1 + 1)
        val pMax = log(cp._2 + 1)
        val p = log(x._2._2 + 1)
        val priceQ = (p - pMin) / (pMax - pMin)
        // sku, price, priceQ
        (x._1, (x._2._2, priceQ))
      }
    }
    */
    priceData

  }
}

