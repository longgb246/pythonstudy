import sys
import numpy as np

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

qieci = {}
# for line in open("common_predicting_model/words_embedding_features/input/embedding.rawdata"):
for line in open("input/embedding.rawdata"):
    segs = line.strip('\n').split('\t')
    qieci[segs[0]] = segs[2].split("")

sales = {}
# for line in open("common_predicting_model/words_embedding_features/input/train_sku.txt"):
for line in open("input/train_sku.txt"):
    segs = line.strip('\n').split('\t')
    #if float(segs[3]) < 10:
    #    continue
    title = qieci[segs[0]]
    num = np.log(float(segs[1]) + 1)
    for w in title:
        if w not in sales:
            sales[w] = []
        sales[w].append(num)

avg = {}
data = []
freq = {}
for key in sales:
    if len(sales[key]) >= 10:
        avg[key] = float(sum(sales[key])) / float(len(sales[key]))
        data.append(float(sum(sales[key])) / float(len(sales[key])))
        freq[key] = len(sales[key])

# f = open("common_predicting_model/words_embedding_features/input/wordsQvalue.txt", "w")
f = open("input/wordsQvalue.txt", "w")
for key in avg:
    qvalue = round((avg[key] - min(data)) / (max(data) - min(data)), 4)
    f.write(key + '\t' + str(qvalue) + '\t' + str(freq[key]) + '\n')
f.close()

printRunTime(t1)