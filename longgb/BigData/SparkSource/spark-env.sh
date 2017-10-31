JAVA_HOME=$JAVA_HOME
HADOOP_HOME=$HADOOP_HOME
HADOOP_CONF_DIR=$HADOOP_CONF_DIR
if [ ! -z $SPARK_EXECUTOR_INSTANCES ];then
    export SPARK_EXECUTOR_INSTANCES=${SPARK_EXECUTOR_INSTANCES:-"10"}
fi
SPARK_LOCAL_IP=`hostname -i`
SPARK_EXECUTOR_CORES=${SPARK_EXECUTOR_CORES:-"2"}
SPARK_EXECUTOR_MEMORY=${SPARK_EXECUTOR_MEMORY:-"8g"}
SPARK_DRIVER_MEMORY=${SPARK_DRIVER_MEMORY:-"4g"}
export SPARK_YARN_QUEUE=$SPARK_YARN_QUEUE

#/data7/spark/hadoop-2.2.0.backup/lib/native
#/data7/spark/hadoop-2.2.0.backup/lib/native:/data7/spark/hadoop-2.2.0.backup/share/hadoop/common/lib/hadoop-lzo-0.4.15.jar
#SPARK_JAVA_OPTS="-verbose:gc -XX:+UseCompressedOops -XX:-PrintGCDetails -XX:+PrintGCTimeStamps -XX:CMSInitiatingOccupancyFraction=60 -Dspark.driver.extraLibraryPath=${HADOOP_HOME}/lib/native -Dspark.executor.extraLibraryPath=${HADOOP_HOME}/lib/native:${HADOOP_HOME}/share/hadoop/common/lib/hadoop-lzo-0.4.15.jar"
SPARK_YARN_JAR="spark.yarn.jar "${SPARK_YARN_JAR_HOME}
SPARK_DRIVER_EXTRALIBRARYPATH="spark.driver.extraLibraryPath "${HADOOP_HOME}/lib/native
SPARK_EXECUTOR_EXTRALIBRARYPATH="spark.executor.extraLibraryPath "${HADOOP_HOME}/lib/native:$HADOOP_LZO
spark_driver_extraLibraryPath=$(sed -n '/spark.driver.extraLibraryPath/p' $SPARK_CONF_DIR/spark-defaults.conf | awk '{print $2}')

#if [ "$spark_driver_extraLibraryPath" != "${HADOOP_HOME}/lib/native" ];
# then
#   sed -i '/spark.executor.extraLibraryPath/d' $SPARK_CONF_DIR/spark-defaults.conf
#   sed -i '/spark.driver.extraLibraryPath/d' $SPARK_CONF_DIR/spark-defaults.conf
  
#   sed -i '$a\\'"$SPARK_DRIVER_EXTRALIBRARYPATH"'' $SPARK_CONF_DIR/spark-defaults.conf
#   sed -i '$a\\'"$SPARK_EXECUTOR_EXTRALIBRARYPATH"'' $SPARK_CONF_DIR/spark-defaults.conf
#fi
export SPARK_LIBRARY_PATH=$HADOOP_HOME/lib/native
export SPARK_CLASSPATH=$SPARK_CLASSPATH:$HADOOP_HOME/lib/native:$HADOOP_LZO:$HIVE_CONF_DIR
