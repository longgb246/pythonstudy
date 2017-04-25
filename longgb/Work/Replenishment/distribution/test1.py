from pyspark import SparkContext,SparkConf,HiveContext


appName = "VLT_OPTIMIZATION"
def main():
    conf = SparkConf().setAppName(appName)
    sc = SparkContext(conf=conf)
    hiveContext = HiveContext(sc)
    hiveContext.sql("create table dev.pySparkTest as select * from dev.tmp_gaoyun_sale limit 100")

if __name__ == "__main__":
    main()




