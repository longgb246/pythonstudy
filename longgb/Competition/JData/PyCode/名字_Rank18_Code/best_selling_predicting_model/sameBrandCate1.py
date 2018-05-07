#encoding:utf-8
import numpy as np
import csv
import Levenshtein
import time
import numpy as np

def printRunTime(t1, name=""):
  d = time.time() - t1
  min_d = np.floor(d / 60)
  sec_d = d % 60
  hor_d = np.floor(min_d / 60)
  if name != "":
    name = " ( " + name + " )"
  if hor_d >0:
    print '[ Run Time{3} ] is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
  else:
    print '[ Run Time{2} ] is : {0} min {1:.4f} s'.format(min_d, sec_d, name)

t1 = time.time()

def is_chinese(uchar):         
  if u'\u4e00' <= uchar<=u'\u9fff':
    return True
  else:
    return False
    

def is_number(uchar):
  if u'\u0030' <= uchar<=u'\u0039':
    return True
  else:
    return False


def is_alphabet(uchar):         
  if (u'\u0041' <= uchar<=u'\u005a') or (u'\u0061' <= uchar<=u'\u007a'):
    return True
  else:
    return False

def sameColor(testColor,color):
  if testColor=='':
    return True
  testColor1=testColor.decode('utf-8').replace(u'色', '')
  color1=color.decode('utf-8').replace(u'色', '')
  res = []                                
  for x in testColor1:                          
    if x in color1 and is_chinese(x):                       
      res.append(x)
  if len(res)>0:
    return True
  else:
    return False


# data=np.genfromtxt("best_selling_predicting_model/input/traindata",delimiter='\t',dtype=np.str)
data=np.genfromtxt("input/traindata",delimiter='\t',dtype=np.str)
# dataTest=np.genfromtxt("best_selling_predicting_model/input/predictdata",delimiter='\t',dtype=np.str)
dataTest=np.genfromtxt("input/predictdata",delimiter='\t',dtype=np.str)

contents=[]
for i in range(len(dataTest)):
  minDis=[]
  minPos=[]
  latestDate='2015-01-01'
  latestDatePos=0
  sales=[]
  itemtypeTest=dataTest[i,4].decode('utf-8')
  for j in range(len(data)):
    itemtype=data[j,6].decode('utf-8')
    if ((dataTest[i,3]==data[j,5]) and (dataTest[i,2]==data[j,4])) or (dataTest[i,1]==data[j,3]):
      #判断同一型号
      if is_number(itemtypeTest[len(itemtypeTest)-1]) and is_alphabet(itemtypeTest[len(itemtypeTest)-2]) and is_chinese(itemtypeTest[len(itemtypeTest)-3]):
         if (is_number(itemtype[len(itemtype)-1]) and is_alphabet(itemtype[len(itemtype)-2]) and is_chinese(itemtype[len(itemtype)-3])) and (itemtypeTest[len(itemtypeTest)-2] ==itemtype[len(itemtype)-2]) and sameColor(dataTest[i,7],data[j,9]):
           minDis.append(Levenshtein.distance(itemtypeTest,itemtype))        
           minPos.append(j)
      else:
        if sameColor(dataTest[i,7],data[j,9]):
          minDis.append(Levenshtein.distance(itemtypeTest,itemtype))
          minPos.append(j)
  if len(minDis)==0:
    continue;
  #分为两种情况：历史有无同种型号销量
  if 0 in minDis:
    for m in range(len(minDis)):
      if (minDis[m]==0) and (data[minPos[m],1]>latestDate) and not(data[minPos[m],1]>='2015-06-01' and data[minPos[m],1]<='2015-06-18') and not(data[minPos[m],1]>='2015-10-31' and data[minPos[m],1]<='2015-11-18') and not(data[minPos[m],1]>='2016-06-01' and data[minPos[m],1]<='2016-06-18') and not(data[minPos[m],1]>='2016-10-31' and data[minPos[m],1]<='2016-11-18') and not(data[minPos[m],1]>='2017-06-01' and data[minPos[m],1]<='2017-06-18') and not(data[minPos[m],1]>='2017-10-31' and data[minPos[m],1]<='2017-11-18'):
        latestDate=data[minPos[m],1]
        latestDatePos=m
    contents.append([dataTest[i,0],dataTest[i,1],itemtypeTest,data[minPos[latestDatePos],2]])
  else:
    for n in range(len(minPos)):
      if not(data[minPos[n],1]>='2015-06-01' and data[minPos[n],1]<='2015-06-18') and not(data[minPos[n],1]>='2015-10-31' and data[minPos[n],1]<='2015-11-18') and not(data[minPos[n],1]>='2016-06-01' and data[minPos[n],1]<='2016-06-18') and not(data[minPos[n],1]>='2016-10-31' and data[minPos[n],1]<='2016-11-18') and not(data[minPos[n],1]>='2017-06-01' and data[minPos[n],1]<='2017-06-18') and not(data[minPos[n],1]>='2017-10-31' and data[minPos[n],1]<='2017-11-18'):
        sales.append(float(data[minPos[n],2]))
    contents.append([dataTest[i,0],dataTest[i,1],itemtypeTest,np.median(sales)])


# pred=np.genfromtxt("output/final.csv",delimiter=',',dtype=np.str)
pred=np.genfromtxt("../common_predicting_model/machine_learning_model/output/final.csv",delimiter=',',dtype=np.str)
# csvfile = file('best_selling_predicting_model/output/final1.csv', 'wb')
csvfile = file('output/final1.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(['SKU_ID', 'forecast_daily_avg_sale'])
#更新预测值
for i in range(len(contents)):
  if float(contents[i][3])>20:
    for j in range(len(pred)):
      if contents[i][0]==pred[j,0] and float(contents[i][3])>float(pred[j,1]):
        pred[j,1]=float(contents[i][3])
        for k in range(len(dataTest)):
          if dataTest[k][0]!=contents[i][0] and dataTest[k][1]==contents[i][1]:
            for m in range(len(pred)):
              if pred[m,0]==dataTest[k][0]:
                pred[m,1]=float(contents[i][3])


writer.writerows(pred)
csvfile.close()
printRunTime(t1)
