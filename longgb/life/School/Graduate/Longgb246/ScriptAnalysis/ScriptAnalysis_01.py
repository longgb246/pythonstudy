# -*- coding:utf-8 -*-
import os
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB


# for num in range(len(filelist)):
#     filetext = filepath + "/" + filelist[num]
#     filename = os.path.basename(filetext)
#     myfile = codecs.open(filetext, 'r', 'utf-8')
#     temp = myfile.readlines()
#     myfile.close()
#     for i in range(0, 100):
#         len_0 = len(temp)
#         seg_list = jieba.cut(','.join(temp[int(i * len_0 / 100):int((i + 1) * len_0 / 100)]), cut_all=False)
#         words = " ".join(seg_list)
#         target_train.append(filename)
#         corpus_train.append(words)


# 处理股票信息，把股价信息变成涨跌信息。1表示涨，0表示跌。

def get_test_data():
    file_name = 'content.txt'
    read_file = read_path + os.sep + file_name
    pd_list = []
    for line in open(read_file):
        tmp = line.split('|')
        if len(tmp) > 4:
            str_tmp = reduce(lambda x, y: x + y, tmp[3:])
        else:
            str_tmp = tmp[3]
        tmp_list = [tmp[0], tmp[1], tmp[2], str_tmp]
        pd_list.append(tmp_list)
    data = pd.DataFrame(pd_list, columns=['date', 'url', 'title', 'content'])
    data['corpus'] = map(lambda x: " ".join(jieba.cut(x[0], cut_all=False)), data.loc[:, ['title']].values)
    corpus_train = data.iloc[range(200), :]
    corpus_test = data.iloc[range(200, 400), :]
    corpus_test.index = range(len(corpus_test))
    return [data, corpus_train, corpus_test]


[data, corpus_train, corpus_test] = get_test_data()


def method_01_load_data():
    '''
    1、使用jieba和sklearn的一些方法，仅用了tf-idf生成的词频矩阵作为分类的输入向量级。用于预测类别
    '''
    # 1、使用 sklearn 能够生成 tf-idf 的词频矩阵
    count_v1 = CountVectorizer()                                            # 该类会将文本中的词语转换为词频矩阵，矩阵元素 a[i][j] 表示j词在i类文本下的词频
    counts_train = count_v1.fit_transform(corpus_train['corpus'])           # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()                                        # 该类会统计每个词语的tf-idf权值
    tfidf_train = transformer.fit(counts_train).transform(counts_train)     # fit_transform是计算tf-idf
    weight_train = tfidf_train.toarray()                                    # weight[i][j],第i个文本，第j个词的tf-idf值
    count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)             # 让两个CountVectorizer共享vocabulary
    counts_test = count_v2.fit_transform(corpus_test['corpus'])             # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()                                        # 该类会统计每个词语的tf-idf权值
    tfidf_test = transformer.fit(counts_train).transform(counts_test)       # fit_transform是计算tf-idf
    weight_test = tfidf_test.toarray()                                      # weight[i][j],第i个文本，第j个词的tf-idf值
    # 2、使用这些矩阵作为输入的特征。再加上label作为分类

    # #---------------------------------------------#
    # knnclf = KNeighborsClassifier()
    # knnclf.fit(weight_train,target_train)
    # knn_pred = knnclf.predict(weight_test)
    # #knn模型
    # #---------------------------------------------#
    # #---------------------------------------------#
    # #svm模型
    # svc = svm.SVC(kernel='linear')
    # svc.fit(weight_train,target_train)
    # svc_pred = svc.predict(weight_test)
    # #---------------------------------------------#
    # #---------------------------------------------#
    # #tree模型
    # tre = tree.DecisionTreeClassifier()
    # tre.fit(weight_train,target_train)
    # tre_pred = tre.predict(weight_test)
    # #---------------------------------------------#
    # #---------------------------------------------#
    # #bayes模型
    # bayes = MultinomialNB(alpha = 0.01)
    # bayes.fit(weight_train,target_train)
    # bayes_pred = bayes.predict(weight_test)
    # #---------------------------------------------#
    #

def method_02():
    # gensim 方法
    pass

def method_03():
    '''
    这个是使用一堆的测量进行一定的度量。但是没有实现，看看效果。
    '''
    def get_term_dict(doc_terms_list):
        term_set_dict = {}
        for doc_terms in doc_terms_list:
            for term in doc_terms:
                term_set_dict[term] = 1
        term_set_list = sorted(term_set_dict.keys())  # term set 排序后，按照索引做出字典
        term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
        return term_set_dict

    def get_class_dict(doc_class_list):
        class_set = sorted(list(set(doc_class_list)))
        class_dict = dict(zip(class_set, range(len(class_set))))
        return class_dict

    def stats_term_df(doc_terms_list, term_dict):
        term_df_dict = {}.fromkeys(term_dict.keys(), 0)
        term_set = []
        for term in term_set:
            for doc_terms in doc_terms_list:
                if term in doc_terms_list:
                    term_df_dict[term] += 1
        return term_df_dict

    def stats_class_df(doc_class_list, class_dict):
        class_df_list = [0] * len(class_dict)
        for doc_class in doc_class_list:
            class_df_list[class_dict[doc_class]] += 1
        return class_df_list

    def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
        term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
        for k in range(len(doc_class_list)):
            class_index = class_dict[doc_class_list[k]]
            doc_terms = doc_terms_list[k]
            for term in set(doc_terms):
                term_index = term_dict[term]
                term_class_df_mat[term_index][class_index] += 1
        return term_class_df_mat

    def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
        A = term_class_df_mat
        B = np.array([(sum(x) - x).tolist() for x in A])
        C = np.tile(class_df_list, (A.shape[0], 1)) - A
        N = sum(class_df_list)
        class_set_size = len(class_df_list)

        term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))
        term_score_max_list = [max(x) for x in term_score_mat]
        term_score_array = np.array(term_score_max_list)
        sorted_term_score_index = term_score_array.argsort()[:: -1]
        term_set_fs = [term_set[index] for index in sorted_term_score_index]

        return term_set_fs

    def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
        A = term_class_df_mat
        B = np.array([(sum(x) - x).tolist() for x in A])
        C = np.tile(class_df_list, (A.shape[0], 1)) - A
        N = sum(class_df_list)
        D = N - A - B - C
        term_df_array = np.sum(A, axis=1)
        class_set_size = len(class_df_list)

        p_t = term_df_array / N
        p_not_t = 1 - p_t
        p_c_t_mat = (A + 1) / (A + B + class_set_size)
        p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
        p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
        p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

        term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
        sorted_term_score_index = term_score_array.argsort()[:: -1]
        term_set_fs = [term_set[index] for index in sorted_term_score_index]

        return term_set_fs

    def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):
        A = term_class_df_mat
        B = np.array([(sum(x) - x).tolist() for x in A])
        C_Total = np.tile(class_df_list, (A.shape[0], 1))
        N = sum(class_df_list)
        C_Total_Not = N - C_Total
        term_set_size = len(term_set)

        p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
        p_t_not_c = (B + 1E-6) / (C_Total_Not + 1E-6 * term_set_size)
        term_score_mat = p_t_c * np.log(p_t_c / p_t_not_c)

        term_score_max_list = [max(x) for x in term_score_mat]
        term_score_array = np.array(term_score_max_list)
        sorted_term_score_index = term_score_array.argsort()[:: -1]
        term_set_fs = [term_set[index] for index in sorted_term_score_index]

        print term_set_fs[:10]
        return term_set_fs

    def feature_selection(doc_terms_list, doc_class_list, fs_method):
        class_dict = get_class_dict(doc_class_list)
        term_dict = get_term_dict(doc_terms_list)
        class_df_list = stats_class_df(doc_class_list, class_dict)
        term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
        term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]
        term_set_fs = []

        if fs_method == 'MI':
            term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
        elif fs_method == 'IG':
            term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
        elif fs_method == 'WLLR':
            term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)

        return term_set_fs

    import os
    import sys

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_files
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    def text_classifly_twang(dataset_dir_name, fs_method, fs_num):
        print 'Loading dataset, 80% for training, 20% for testing...'
        movie_reviews = load_files(dataset_dir_name)
        doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(
            movie_reviews.data, movie_reviews.target, test_size=0.2, random_state=0)
        print 'Feature selection...'
        print 'fs method:' + fs_method, 'fs num:' + str(fs_num)
        vectorizer = CountVectorizer(binary=True)
        word_tokenizer = vectorizer.build_tokenizer()
        doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]
        term_set_fs = feature_selection.feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)[
                      :fs_num]
        print 'Building VSM model...'
        term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))
        vectorizer.fixed_vocabulary = True
        vectorizer.vocabulary_ = term_dict
        doc_train_vec = vectorizer.fit_transform(doc_str_list_train)
        doc_test_vec = vectorizer.transform(doc_str_list_test)
        clf = MultinomialNB().fit(doc_train_vec, doc_class_list_train)  # 调用MultinomialNB分类器
        doc_test_predicted = clf.predict(doc_test_vec)
        acc = np.mean(doc_test_predicted == doc_class_list_test)
        print 'Accuracy: ', acc
        return acc

    dataset_dir_name = sys.argv[1]
    fs_method_list = ['IG', 'MI', 'WLLR']
    fs_num_list = range(25000, 35000, 1000)
    acc_dict = {}

    for fs_method in fs_method_list:
        acc_list = []
        for fs_num in fs_num_list:
            acc = text_classifly_twang(dataset_dir_name, fs_method, fs_num)
            acc_list.append(acc)
        acc_dict[fs_method] = acc_list
        print 'fs method:', acc_dict[fs_method]

    for fs_method in fs_method_list:
        plt.plot(fs_num_list, acc_dict[fs_method], '--^', label=fs_method)
        plt.title('feature  selection')
        plt.xlabel('fs num')
        plt.ylabel('accuracy')
        plt.ylim((0.82, 0.86))

    plt.legend(loc='upper left', numpoints=1)
    plt.show()
    pass

def method_04():
    pass


# =================================================================
# =                             参数配置                          =
# =================================================================
read_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Now_Data\test\600000'
save_path = r'F:\Learning\School_Master\Graduate\Codes_Data\Now_Data\test\600000'

if __name__ == '__main__':
    pass
