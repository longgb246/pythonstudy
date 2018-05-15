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

object ATestPromotionExtractor {
  def extract(sc: SparkContext, projectPath: String,
    skuFirstDt: RDD[(String, String)],
    skuPrice: RDD[(String, List[Double])]) = {

    val rulePath = projectPath + "/data/B_promotion_rule.csv"
    val ruleData = GBKtoUtf8.transfer(sc, rulePath).distinct.map(
      _.split("\t")
    ).filter(x => x.length > 11).filter( x =>
      x(11) != "" && x(11) != "-1"
    ).map { x =>
      val promotion_id = x(0)
      val topdiscount = x(9)
      val fixedprice = x(10)
      val discount = x(11).trim.toDouble
      //val discount_num = x(15)
      //val neednum = x(16)
      //val rate = x(17)
      (promotion_id, discount)
    }.distinct

    val coveragePath = projectPath + "/data/B_promotion_coverage.csv"
    val coverageData = GBKtoUtf8.transfer(sc, coveragePath).distinct.map(
      _.split("\t")
    ).filter( x =>
      x(6) != '1' && x.length == 9
    ).map { x =>
      val promotion_id = x(0)
      val thirdCate = x(5)
      val sku_id = x(8).trim
      (promotion_id, sku_id)
    }.distinct

    val recordPath = projectPath + "/data/B_promotion_record.csv"
    val recordData = GBKtoUtf8.transfer(sc, recordPath).distinct.map(
      _.split("\t")
    ).map { x =>
      val promotionId = x(0)
      val promotionType = x(2).trim.toInt
      val timeBegin = x(4).split(" ")(0).trim
      val timeEnd = x(5).split(" ")(0).trim
      (promotionId, (timeBegin, timeEnd))
    }.distinct

    val skuPromotion = ruleData.join(recordData).map { x =>
      (x._1, (x._2._1, x._2._2._1, x._2._2._2))
    }.join(coverageData).map { x =>
      val sku = x._2._2
      (sku, x._2._1)
    }

    val skuDiscount = skuFirstDt.join(skuPromotion).map { x =>
      val sku = x._1
      val firstDt = x._2._1
      val promotion = x._2._2
      val format = new SimpleDateFormat("yyyy-MM-dd")
      val firstDate = format.parse(firstDt)
      var discount = 0.0
      var count = 0
      Range(0, 7).foreach { i =>
        val dt = {
          val cal = Calendar.getInstance
          cal.setTime(firstDate)
          cal.add(Calendar.DATE, i)
          format.format(cal.getTime)
        }
        if(dt >= promotion._2 && dt <= promotion._3) {
          discount += promotion._1
          count += 1
        }
      }
      val dc = if(count > 0) (discount / count) else 0
      (sku, dc)
    }.groupByKey().map { x =>
      val v = x._2.toList
      (x._1, v.sum / v.length)
    }

    val skuDisRate = skuDiscount.join(skuPrice).map { x=>
      val price = x._2._2(0)
      val discountQValue = if (x._2._1 < price) {
        x._2._1 / price
      } else 1.0
      (x._1, List(discountQValue))
    }

    skuDisRate
  }
}

