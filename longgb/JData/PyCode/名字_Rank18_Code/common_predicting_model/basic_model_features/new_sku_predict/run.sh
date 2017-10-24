spark-submit \
  --class "com.jd.sale_predict.feature.FeatureExtractor" \
  --num-executors 100 \
  target/scala-2.11/sale-predict_2.11-1.0.jar \
  $@
