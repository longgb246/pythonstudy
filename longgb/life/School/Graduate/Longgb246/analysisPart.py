#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.neighbors import KNeighborsClassifier



def cleanDate(x):
    try:
        tmp_str = pd.to_datetime(x)
        return tmp_str
    except:
        return np.nan


def readPartData():
    file_path = read_path + os.sep + 'sha_tmp_700.csv'
    text_data = pd.read_csv(file_path)
    text_data['date'] = map(cleanDate, text_data['date'].values)
    text_data.dropna(inplace=True)
    text_data['date'] = pd.to_datetime(text_data['date'])
    text_data.index = range(len(text_data))
    text_data['com_code'] = text_data['com_code'].astype(int)
    text_data['com_code'] = text_data['com_code'].astype(str)
    return text_data


def readFinance():
    file = read_path_root + os.sep + 'Finance'
    read_finance = pd.DataFrame()
    for read_dir in read_dirs:
        read_finance_path = file + os.sep + 'stock_price_{0}.csv'.format(read_dir)
        tmp_finance = pd.read_csv(read_finance_path)
        tmp_finance = tmp_finance.loc[:,['date', 'open', 'close', 'code']]
        read_finance = pd.concat([read_finance, tmp_finance])
    read_finance['change'] = read_finance['close'] - read_finance['open']
    read_finance['change_flag'] = map(lambda x: 1 if x > 0 else 0,read_finance['change'].values)
    read_finance = read_finance.rename(columns={'code':'com_code'})
    read_finance['date'] = pd.to_datetime(read_finance['date'])
    read_finance['com_code'] = read_finance['com_code'].astype(str)
    return read_finance


def getTrainTest(corpus_vec):
    sample_index = range(len(corpus_vec))
    np.random.shuffle(sample_index)
    index_test = sample_index[:int(len(corpus_vec) * 0.2)]
    index_train = list(set(sample_index) - set(index_test))
    corpus_train = corpus_vec.loc[index_train, ['abstract', 'change_flag']]
    corpus_train.index = range(len(corpus_train))
    corpus_test = corpus_vec.loc[index_test, ['abstract', 'change_flag']]
    corpus_test.index = range(len(corpus_test))
    train_vec = corpus_train['abstract'].tolist()
    train_flag = corpus_train['change_flag'].tolist()
    test_vec = corpus_test['abstract'].tolist()
    test_flag = corpus_test['change_flag'].tolist()
    return [train_vec, train_flag, test_vec, test_flag]


def getVector(train_vec, train_flag, test_vec, test_flag):
    count_v1 = CountVectorizer()                                            # 该类会将文本中的词语转换为词频矩阵，矩阵元素 a[i][j] 表示j词在i类文本下的词频
    counts_train = count_v1.fit_transform(train_vec)                        # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()                                        # 该类会统计每个词语的tf-idf权值
    tfidf_train = transformer.fit(counts_train).transform(counts_train)     # fit_transform是计算tf-idf
    weight_train = tfidf_train.toarray()                                    # weight[i][j],第i个文本，第j个词的tf-idf值
    count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)             # 让两个CountVectorizer共享vocabulary
    counts_test = count_v2.fit_transform(test_vec)                          # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()                                        # 该类会统计每个词语的tf-idf权值
    tfidf_test = transformer.fit(counts_train).transform(counts_test)       # fit_transform是计算tf-idf
    weight_test = tfidf_test.toarray()                                      # weight[i][j],第i个文本，第j个词的tf-idf值

    knnclf = KNeighborsClassifier()
    knnclf.fit(weight_train,train_flag)
    knn_pred = knnclf.predict(weight_test)
    print knn_pred



# =================================================================
# =                             参数配置                          =
# =================================================================
read_path_root = r'F:\Learning\School_Master\Graduate\Codes_Data'
read_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Now_Data\integratedData'
save_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Now_Data\integratedData'

read_dirs = ['sha', 'shb', 'szb', 'szcy', 'szz', 'szzx']


if __name__ == '__main__':
    print 'read_finance...'
    read_finance = readFinance()
    print 'text_data...'
    text_data = readPartData()
    print 'merge...'
    corpus = text_data.merge(read_finance, on=['date','com_code'])
    corpus['abstract'] = corpus['title'] + " " + corpus['cont_abstract']
    corpus_vec = corpus.loc[:,['date','com_code','abstract','change_flag']]
    [train_vec, train_flag, test_vec, test_flag] = getTrainTest(corpus_vec)
    getVector(train_vec, train_flag, test_vec, test_flag)
    pass


# 之后把 read_finance 的数据再按照新闻的时间进行扩展。取更多的 flag 点
