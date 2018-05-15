#!/bin/bash

# basic features
#cd common_predicting_model/basic_model_features/new_sku_predict
#sh build.sh
#sh run.sh

# combine features
python common_predicting_model/words_embedding_features/words_qvalue.py
python common_predicting_model/words_embedding_features/qvalue_sku.py
# machine learning
python common_predicting_model/machine_learning_model/multiStackingRealize.py
# postprocessing
python best_selling_predicting_model/sameBrandCate1.py

python best_selling_predicting_model/sameBrandCate2.py
