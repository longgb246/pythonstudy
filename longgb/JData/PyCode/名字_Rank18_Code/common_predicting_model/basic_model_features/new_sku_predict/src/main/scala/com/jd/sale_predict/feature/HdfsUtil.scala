package com.jd.sale_predict.feature

object HdfsUtil {
  def deleteHdfsPath(filePath: String) = {
    val prefix = "hdfs://ns3/user/jd_ad/lisai/sale_predict/"
    if (!filePath.startsWith(prefix)) {
      System.err.println(
        s"Only path that starts with '${prefix}' " +
        "can be delete FOR SAFETY REASON.")
    } else {
      val hadoopConf = new org.apache.hadoop.conf.Configuration()
      val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
      try {
        hdfs.delete(new org.apache.hadoop.fs.Path(filePath), true)
      } catch {
        case _ : Throwable => { }
      }
    }
  }
}
