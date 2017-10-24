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


object ATestFeatureExtractor {
  def extract(sc: SparkContext, projectPath: String, outputPath: String) = {

    // 首次可售 RDD: (sku, firstDt)
    val skuFirstDt = ATestVendExtractor.extractFirstDt(sc, projectPath)
    // 可售状态 RDD: (sku, Iterable(dt))
    //val skuVendData = ATestVendExtractor.extract(skuFirstDt)
    // 每天销量 RDD: ((sku, dt), sale)
    val saleData = OrderExtractor.extract(sc, projectPath)
    // 每日pv、uv与停留时长 RDD: ((sku, dt), (pv, uv, stay_time)
    val visitData = VisitExtractor.extract(sc, projectPath)
    // sku信息 RDD: (sku, (brand, cate, type, sku_name))
    val infoFile = "B_sku_basic_info_final.csv"
    val newInfoData = BasicInfoExtractor.extract(sc, projectPath, infoFile, ",")
    // 价格 RDD: (sku, (price, priceQValue))
    val skuPrice = ATestPriceExtractor.extract(sc, projectPath, outputPath, newInfoData)

    // 促销力度 RDD: (sku, discount)
    val skuPromotion = ATestPromotionExtractor.extract(
      sc, projectPath, skuFirstDt, skuPrice)
    // sku新品前七天 RDD: (sku, brand, cate, dt)
    val oldInfoFile = "{train_sku_basic_info.csv,sku_basic_info.csv}"
    val oldInfoData = BasicInfoExtractor.extract(sc, projectPath, oldInfoFile, "\t")
    val infoData = newInfoData.map( x =>
      (x._1, (x._2, 1))
    ).union(
      oldInfoData.map(x => (x._1, (x._2, 2)))
    ).groupByKey().map { x =>
      val v = x._2.toList
      var r = (x._1, v(0)._1)
      if (v.length > 1) {
        v.foreach { i =>
          if (i._2 == 1) r = (x._1, i._1)
        }
      }
      r
    }.distinct

    val cid1Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid1", 7)
    val cid1Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid1", 30)
    val cid2Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid2", 7)
    val cid2Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid2", 30)
    val cid3Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid3", 7)
    val cid3Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid3", 30)
    val brandSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand", 7)
    val brandThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand", 30)
    val itemTypeSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "itemType", 7)
    val itemTypeThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "itemType", 30)
    val spuSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "spu", 7)
    val spuThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "spu", 30)
    val brandC1Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid1", 7)
    val brandC1Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid1", 30)
    val brandC2Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid2", 7)
    val brandC2Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid2", 30)
    val brandC3Seven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid3", 7)
    val brandC3Thirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid3", 30)
    val brandITSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-itemType", 7)
    val brandITThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-itemType", 30)
    val brandC3ITSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid3-iT", 7)
    val brandC3ITThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "brand-cid3-iT", 30)
    val c3ITSeven = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid3-iT", 7)
    val c3ITThirty = ATestSaleQValueExtractor.extract(skuFirstDt, infoData, saleData, "cid3-iT", 30)

    val brandCateDt = ATestQValueExtractor.extractBrandCateDt(skuFirstDt, newInfoData)
    val brandVisitValue = QValueExtractor.extractBrandVisitValue(brandCateDt,
      visitData, oldInfoData)

    // 品牌销量中位数
    val brandMiddleSale = QValueExtractor.extractBrandMiddleSale(brandCateDt,
      saleData, oldInfoData)
    val oldSkuVendData = VendExtractor.extract(sc, projectPath)
    // 标签：新品销量  RDD: (sku, (firstDt, label))
    val oldSkuLabel = LabelExtractor.extractLabel(oldSkuVendData, saleData).leftOuterJoin(infoData).filter { x =>
      val info = x._2._2.getOrElse(Info("","","","","","","","",""))
      !info.skuName.contains("【O仓】")
    }.map( x => (x._1, x._2._1))
    val newSkuLabel = skuFirstDt.map( x => (x._1, (x._2, -1.0)))
    val skuLabel = oldSkuLabel.union(newSkuLabel).distinct

    val cid1Label = LabelQValueExtractor.extract(skuLabel, infoData, "cid1")
    val cid2Label = LabelQValueExtractor.extract(skuLabel, infoData, "cid2")
    val cid3Label = LabelQValueExtractor.extract(skuLabel, infoData, "cid3")
    val brandLabel = LabelQValueExtractor.extract(skuLabel, infoData, "brand")
    val itemTypeLabel = LabelQValueExtractor.extract(skuLabel, infoData, "itemType")
    val spuLabel = LabelQValueExtractor.extract(skuLabel, infoData, "spu")
    val brandC1Label = LabelQValueExtractor.extract(skuLabel, infoData, "brand-cid1")
    val brandC2Label = LabelQValueExtractor.extract(skuLabel, infoData, "brand-cid2")
    val brandC3Label = LabelQValueExtractor.extract(skuLabel, infoData, "brand-cid3")
    val brandITLabel = LabelQValueExtractor.extract(skuLabel, infoData, "brand-itemType")
    val brandC3ITLabel = LabelQValueExtractor.extract(skuLabel, infoData, "brand-cid3-iT")
    val c3ITLabel = LabelQValueExtractor.extract(skuLabel, infoData, "cid3-iT")
 
    // fea: label, price, promotion, brandQ, cateQ, sameBC
     val feaList = List(skuPrice, cid1Seven, cid1Thirty, cid2Seven, cid2Thirty,
      cid3Seven, cid3Thirty, brandSeven, brandThirty, itemTypeSeven, itemTypeThirty,
      spuSeven, spuThirty, brandC1Seven, brandC1Thirty, brandC2Seven, brandC2Thirty,
      brandC3Seven, brandC3Thirty, brandITSeven, brandITThirty, brandC3ITSeven, brandC3ITThirty,
      c3ITSeven, c3ITThirty, cid1Label, cid2Label, cid3Label, brandLabel, itemTypeLabel,
      spuLabel, brandC1Label, brandC2Label, brandC3Label, brandITLabel, brandC3ITLabel,
      c3ITLabel, skuPromotion
    )

    var fea = skuFirstDt.join(newInfoData).map { x =>
      val skuInfo: List[Any] = List(x._2._1,x._2._2.spu, x._2._2.brand,
        x._2._2.cid3, x._2._2.itemType, x._2._2.skuName, x._2._2.shelvesDt,x._2._2.colour
      )
      (x._1, skuInfo)
    }
    var i = 0
    feaList.foreach { f =>
      fea = fea.leftOuterJoin(f).map { x =>
        val feaValue = x._2._2.getOrElse(List(0.0))
        val valueList = x._2._1 ::: feaValue
        (x._1, valueList)
      }
      if (i > 25) {
        println("fuck:" + i + fea.count())
      }
      i = i + 1
    }
    fea = fea.leftOuterJoin(brandVisitValue).map { x =>
      val feaValue = x._2._2.getOrElse(List(0.0, 0.0, 0.0))
      val valueList = x._2._1 ::: feaValue
      (x._1, valueList)
    }
    fea = fea.leftOuterJoin(brandMiddleSale).map { x =>
      val feaValue = x._2._2.getOrElse(List(0.0))
      val valueList = x._2._1 ::: feaValue
      (x._1, valueList)
    }
    // sku, label, price, disCount, brandQValue, cateQValue, sameBrandCate
    val result = fea.map { x =>
      (x._1 :: x._2).mkString("\t")
    }   

    val resultPath = outputPath + "/predictdata/"
    HdfsUtil.deleteHdfsPath(resultPath)
    result.coalesce(1).saveAsTextFile(resultPath)
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Sale Predict")
    val sc = new SparkContext(conf)

    var projectPath = "sale_predict/"
    var outputPath = "lvlei/sale_predict/"
    var today = new Date()
    if (args.length >= 1) {
      projectPath = args(0)
    }
    if (args.length >= 2) {
      outputPath = args(1)
    }
    extract(sc, projectPath, outputPath)
  }
}
