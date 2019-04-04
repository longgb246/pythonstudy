#-*- coding:utf-8 -*-
##### Enter the docker
# xxxvl start -i bdp-docker.xxx.com:5000/wise_mart_cmo_ipc -o='--net=host' -I bash
# xxxvl start -m /data0/spark:/data0/spark:ro -m /data0/cmo_ipc:/data0/cmo_ipc:rw -i bdp-docker.xxx.com:5000/wise_mart_cmo_ipc -o='--net=host' -I bash
# 如果你有权限读写 /data0/com_ipc, 就 -m /data0/cmo_ipc:/data0/cmo_ipc:rw
# 不是 -v 是 -m, 还有这个 bash 后面是不能跟 xxxvl 的选项的  (xxxvl start -h 输出)
##### Use the Pyspark
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cmo_ipc:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --files $HIVE_CONF_DIR/hive-site.xml


from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
a_sp = spark.createDataFrame([[2,3,4,4], [4,4,4,4]], list('abcd'))
a_sp.columns
a_sp.printSchema()
mm = a_sp.schema
str(mm)


a_sp.select((F.col('a')*1.0/F.col('b'))).show()
is_211_sp = spark.read.csv('hdfs://ns15/user/cmo_ipc/longguangbin/work/dev_lgb_fullstock_timinglimit_order_waybill_is211_spark', header=True)
is_211_sp.show()


### isc
# "spark-submit --master yarn " \
# " --deploy-mode cluster" \
# " --num-executors 20" \
# " --executor-memory 10g" \
# " --executor-cores 4" \
# " --driver-memory 10g" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_isc:latest" \
# " --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_isc:latest" \
# " --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server" \
# " --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server " \
# " --files /software/conf/10k/mart_isc/bdp_jmart_isc_union.bdp_jmart_isc_dev/hive_conf/hive-site.xml " \
# " --conf spark.pyspark.python=python2" \
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_isc:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_isc:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --files /software/conf/10k/mart_isc/bdp_jmart_isc_union.bdp_jmart_isc_dev/hive_conf/hive-site.xml


### cib
# "spark-submit --master yarn --deploy-mode cluster" \
# " --num-executors 20" \
# " --executor-memory 10g" \
# " --executor-cores 4" \
# " --driver-memory 10g" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest" \
# " --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest" \
# " --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server" \
# " --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server" \
# " --py-files {zip_files}" \
# " {pyfile} "
#### xxxvl start -m /data0/mart_cib:/data0/mart_cib:rw -i bdp-docker.xxx.com:5000/wise_mart_cib:latest -o='--net=host' -I bash
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server

# sys.path.append('/home/mart_cib/longguangbin/saas/dev_promotion_new/tt1')


### bca
# "spark-submit --master yarn --deploy-mode cluster" \
# " --num-executors 40" \
# " --executor-memory 20g" \
# " --executor-cores 5" \
# " --driver-memory 20g" \
# " --driver-cores 12" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer" \
# " --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest" \
# " --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest" \
# " --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server" \
# " --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server" \
# " --py-files {zip_files}" \
# " {pyfile}  {args}"
#### xxxvl start -m /data0/mart_bca:/data0/mart_bca:rw -i bdp-docker.xxx.com:5000/wise_mart_bca:latest -o='--net=host' -I bash
# pyspark --master yarn \
# --num-executors 10 \
# --executor-memory 10g \
# --executor-cores 4 \
# --driver-memory 10g \
# --conf spark.driver.maxResultSize=20g \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
# --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.xxx.com:5000/wise_mart_cib:latest \
# --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server \
# --conf spark.executorEnv.LD_LIBRARY_PATH=/software/servers/hadoop-2.7.1/lib/native:/software/servers/xxxk1.8.0_121/jre/lib/amd64/server

