import numpy as np
import csv 
import sklearn
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
clfs=[
    GradientBoostingRegressor(
      loss='huber'
      , learning_rate=0.1
      , n_estimators=100
      , subsample=1
      , min_samples_split=2
      , min_samples_leaf=1
      , max_depth=3
      , init=None
      , random_state=None
      , max_features=None
      , alpha=0.9
      , verbose=0
      , max_leaf_nodes=None
      , warm_start=False
      ),
    xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1),
    RandomForestRegressor(),
    ExtraTreesRegressor(),
]
data=np.genfromtxt("common_predicting_model/words_embedding_features/output/f1_trainset.data",delimiter='\t',dtype=np.float32)
train_feat=data[:,11:1292]
train_id=np.log(data[:,2] + 1)

dataTest=np.genfromtxt("common_predicting_model/words_embedding_features/output/f1_predictset.data",delimiter='\t',dtype=np.float32)
test_feat=dataTest[:,10:1291]

blend_train = np.zeros((train_feat.shape[0], len(clfs)))
blend_test = np.zeros((test_feat.shape[0], len(clfs)))
kf=KFold(n_splits=5,random_state=2017)
for j, clf in enumerate(clfs):
    print 'Training classifier [%s]' % (j)
    blend_test_j = np.zeros((test_feat.shape[0], 5)) 
    for i, (train_index, cv_index) in enumerate(kf.split(train_feat)):
        print 'Fold [%s]' % (i)
        X_train = train_feat[train_index]
        Y_train = train_id[train_index]
        X_cv = train_feat[cv_index]
        clf.fit(X_train, Y_train)
        blend_train[cv_index, j] = clf.predict(X_cv)
        blend_test_j[:, i] = clf.predict(test_feat)
    blend_test[:, j] = blend_test_j.mean(1)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(blend_train, train_id)
pred = model_xgb.predict(blend_test)
result=np.exp(pred)-1
pred=result

sku_ids=np.genfromtxt("common_predicting_model/words_embedding_features/output/f1_predictset.data",delimiter='\t',dtype=np.str)
contents=[]
for i in range(pred.shape[0]):
    print sku_ids[i,0],pred[i]
    contents.append([sku_ids[i,0],pred[i]])

csvfile = file('common_predicting_model/machine_learning_model/output/final.csv', 'wb')
writer = csv.writer(csvfile) 
writer.writerows(contents)
csvfile.close()

