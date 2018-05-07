#! /bin/sh

loginfo() { echo "[INFO] $@"; }
logerror() { echo "[ERROR] $@" 1>&2; }

################################################################################
loginfo "sbt package"
sbt package
if [ $? -ne 0 ]; then
  logerror "sbt package failed."
  exit 1
fi

################################################################################
loginfo "cp package jar to output"
rm -rf output
mkdir output
mkdir output/lib

PACKAGE_JAR_NAME=sale-predict_2.11-1.0.jar
PACKAGE_JAR=./target/scala-2.11/${PACKAGE_JAR_NAME}

cp ${PACKAGE_JAR} ./output/lib/
if [ $? -ne 0 ]; then
  logerror "mv package jar failed."
  exit 1
fi 

################################################################################
loginfo "get git log info"
git log --pretty=oneline -1 >> ./output/log.txt
if [ $? -ne 0 ]; then
    logerror "get git log info failed"
fi

################################################################################
loginfo "zip output"
rm output.zip
zip -r output.zip ./output/*

OUTPUT_ZIP_NAME=output.zip
OUTPUT_ZIP=./output.zip
MD5=$(md5sum ${OUTPUT_ZIP} | awk '{print $1}')
cp ${OUTPUT_ZIP} "./output/${OUTPUT_ZIP_NAME%.zip}.${MD5}.zip"
if [ $? -ne 0 ]; then
    logerror "zip and cp output failed"
    exit 1
fi


loginfo "build success"

