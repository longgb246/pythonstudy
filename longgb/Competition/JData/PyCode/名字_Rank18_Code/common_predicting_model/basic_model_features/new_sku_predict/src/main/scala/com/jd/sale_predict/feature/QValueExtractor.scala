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

object QValueExtractor {
  def extractBrandCateDt(skuFirstDt: RDD[(String, String)],
    infoData: RDD[(String, Info)]) = {
    val brandCateDt = skuFirstDt.join(infoData).flatMap { x =>
      val sku = x._1
      val firstDt = x._2._1
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

  def extractBrandQValue(brandCateDt: RDD[(String, String, String, String)],
    saleData: RDD[((String, String), Double)],
    infoData: RDD[(String, Info)]) = {
    val brandSale = saleData.map( x =>
      (x._1._1, (x._1._2, x._2))
    ).join(infoData).map { x =>
      val sku = x._1
      val brand = x._2._2.brand
      val dt = x._2._1._1
      val sale = x._2._1._2
      ((brand, dt), (sale, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map { x =>
      (x._1._2, (x._1._1, x._2._1 / x._2._2))
    }.groupByKey().flatMap { x =>
      val vList = x._2.toList.map(i => (i._1, log(i._2 + 1)))
      val saleList = vList.map(_._2)
      val minSale = saleList.min
      val maxSale = saleList.max
      val minusSale = maxSale - minSale
      vList.map { i =>
        val saleQ = if(minusSale > 0) ((i._2 - minSale) / minusSale) else 0.0
        ((i._1, x._1), saleQ)
      }
    }

    val skuBrandQValue = brandCateDt.map { x =>
      ((x._2, x._4), x._1)
    }.join(brandSale).map { x =>
      val brand = x._1._1
      val dt = x._1._2
      val sku = x._2._1
      val saleQ = x._2._2
      (sku, (saleQ, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map( x =>
      (x._1, List(x._2._1 / x._2._2))
    )
    skuBrandQValue
  }


  def extractBrandMiddleSale(brandCateDt: RDD[(String, String, String, String)],
    saleData: RDD[((String, String), Double)],
    infoData: RDD[(String, Info)]) = {
    
    val newbrandCateDt = brandCateDt.map { x =>
      ((x._1, x._2, x._4), 1)
    }
    
    val brandSale = saleData.map( x =>
      (x._1._1, (x._1._2, x._2))
    ).join(infoData).map { x =>
      val sku = x._1
      val brand = x._2._2.brand
      val dt = x._2._1._1
      val sale = x._2._1._2
      ()
      ((sku, brand, dt),  sale)
    }.join(newbrandCateDt).map { x =>
      val sku = x._1._1
      val brand = x._1._2
      val sale = x._2._1
      ((sku, brand), (sale, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map{ x =>
      (x._1._2, (x._2._1 / x._2._2))
    }.groupByKey().map { x =>
      val vList = x._2.toList.sortWith(_ < _)
      (x._1, vList(vList.length / 2))
    }

    val skuBrandMiddleSale = brandCateDt.map { x =>
      (x._2, x._1)
    }.distinct.join(brandSale).map { x =>
      val sku = x._2._1
      val brand = x._1
      val middleSale = x._2._2
      (sku, List(middleSale))
    }
    skuBrandMiddleSale
  }


  def extractBrandVisitValue(brandCateDt: RDD[(String, String, String, String)],
    visitData: RDD[((String, String), (Double, Double, Double))],
    infoData: RDD[(String, Info)]) = { 
    val brandVisit = visitData.map( x =>
      (x._1._1, (x._1._2, x._2._1, x._2._2, x._2._3))
    ).join(infoData).map { x =>
      val sku = x._1
      val brand = x._2._2.brand
      val dt = x._2._1._1
      val pv = x._2._1._2
      val uv = x._2._1._3
      val stay_time = x._2._1._4
      ((brand, dt), (pv, uv, stay_time, 1)) 
    }.reduceByKey((a, b) => 
      (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)).map { x =>
      (x._1._2, (x._1._1, x._2._1 / x._2._4, x._2._2 / x._2._4, x._2._3 / x._2._4))
    }.groupByKey().flatMap { x =>
      val vList = x._2.toList.map(i => 
        (i._1, log(i._2 + 1), log(i._3 + 1), log(i._4 + 1)))
      val pvList = vList.map(_._2)
      val uvList = vList.map(_._3)
      val stList = vList.map(_._4)
      val minPV = pvList.min
      val maxPV = pvList.max
      val minUV = uvList.min
      val maxUV = uvList.max
      val minST = stList.min
      val maxST = stList.max
      val minusPV = maxPV - minPV
      val minusUV = maxUV - minUV
      val minusST = maxST - minST
      vList.map { i =>
        val pvQ = if(minusPV > 0) ((i._2 - minPV) / minusPV) else 0.0
        val uvQ = if(minusUV > 0) ((i._3 - minUV) / minusUV) else 0.0
        val stQ = if(minusST > 0) ((i._4 - minST) / minusST) else 0.0
        ((i._1, x._1), (pvQ, uvQ, stQ))
      }
    }

    val skuBrandVisitValue = brandCateDt.map { x =>
      ((x._2, x._4), x._1)
    }.join(brandVisit).map { x =>
      val brand = x._1._1
      val dt = x._1._2
      val sku = x._2._1
      val pvQ = x._2._2._1
      val uvQ = x._2._2._2
      val stQ = x._2._2._3
      (sku, (pvQ, uvQ, stQ, 1)) 
    }.reduceByKey((a, b) => 
      (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)).map( x =>
      (x._1, List(x._2._1 / x._2._4, x._2._2 / x._2._4, x._2._3 / x._2._4))
    )
    skuBrandVisitValue
  }


  def extractCateQValue(brandCateDt: RDD[(String, String, String, String)],
    saleData: RDD[((String, String), Double)],
    infoData: RDD[(String, (String, String, String ,String))]) = {
    val cateSale = saleData.map(x =>
      (x._1._1, (x._1._2, x._2))
    ).join(infoData).map { x =>
      val sku = x._1
      val cate = x._2._2._2
      val dt = x._2._1._1
      val sale = x._2._1._2
      ((cate, dt), (sale, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map { x =>
      (x._1._2, (x._1._1, x._2._1 / x._2._2))
    }.groupByKey().flatMap { x =>
      val vList = x._2.toList.map(i => (i._1, log(i._2 + 1)))
      val saleList = vList.map(_._2)
      val minSale = saleList.min
      val maxSale = saleList.max
      val minusSale = maxSale - minSale
      vList.map { i =>
        val saleQ = if(minusSale > 0) ((i._2 - minSale) / minusSale) else 0.0
        ((i._1, x._1), saleQ)
      }
    }

    val skuCateQValue = brandCateDt.map(x =>
      ((x._3, x._4), x._1)
    ).join(cateSale).map { x =>
      val cate = x._1._1
      val dt = x._1._2
      val sku = x._2._1
      val saleQ = x._2._2
      (sku, (saleQ, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).map( x =>
      (x._1, x._2._1 / x._2._2)
    )
    skuCateQValue
  }

}

