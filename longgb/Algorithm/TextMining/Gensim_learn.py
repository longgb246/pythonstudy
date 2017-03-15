#-*- coding:utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from six import iteritems
import os
from gensim import corpora, models, similarities
import gensim
import bz2


def l1_Corpora():
    # 用到了 bag-of-words ?
    # 原始文本
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    # -------------------------------- 标记文本 --------------------------------
    stoplist = set('for a of the and to in'.split())                             # 设置停用词
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]         # 小写。去掉停用词。代码风骚！
    frequency = defaultdict(int)
    # 统计词频
    for text in texts:
        for token in text:
            frequency[token] += 1
    # 注意：仅仅留下词频大于 1 的词
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    # 你的方式处理的文件可能会有所不同；在这里，我只把空格分割，其次是小写字母每个字。事实上，我使用这个特定的（简单的和低效的）设置模拟迪尔韦斯特等人所做的实验。
    dictionary = corpora.Dictionary(texts)                                         # 形成id的字典，单词：id
    dictionary.save(work_path + os.sep + 'deerwester.dict')
    # print dictionary.token2id
    # new_doc = "Human computer interaction"
    # new_vec = dictionary.doc2bow(new_doc.lower().split())                        # 使用之前的dictionary，将新的文本形成词向量，是id：词频
    # -------------------------------- 语料库 --------------------------------
    corpus = [dictionary.doc2bow(text) for text in texts]                         # 这个就是语料库了吗
    # corpora.MmCorpus.serialize(work_path + os.sep + 'deerwester.mm', corpus)     # 保存，为了之后使用，序列化
    # corpus2 = corpora.MmCorpus(work_path + os.sep + 'deerwester.mm')             # 序列化的读取
    # 迭代的建立语料库
    stoplist = set('for a of the and to in'.split())
    dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)                                  # 移除停用词和只出现过一次的
    dictionary.compactify()                                                        # 使得被移除的词的id被移除


def l2_Topics():
    dictionary = corpora.Dictionary.load(work_path + os.sep + 'deerwester.dict')
    corpus = corpora.MmCorpus(work_path + os.sep + 'deerwester.mm')
    tfidf = models.TfidfModel(corpus)                       # step 1 : 初始化一个model TF-IDF，这里用的是TF-IDF。其他的模型，如： Latent Semantic Analysis 和 Latent Dirichlet Allocation。在弄。
    doc_bow = [(0, 1), (1, 1)]
    print(tfidf[doc_bow])                                  # step 2 -- use the model to transform vectors
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)           # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf]                          # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    lsi.print_topics(2)
    for doc in corpus_lsi:                                 # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
        print(doc)
    # [(0, -0.066), (1, 0.520)]     # "Human machine interface for lab abc computer applications"
    # [(0, -0.197), (1, 0.761)]     # "A survey of user opinion of computer system response time"
    # [(0, -0.090), (1, 0.724)]     # "The EPS user interface management system"
    # [(0, -0.076), (1, 0.632)]     # "System and human system engineering testing of EPS"
    # [(0, -0.102), (1, 0.574)]     # "Relation of user perceived response time to error measurement"
    # [(0, -0.703), (1, -0.161)]    # "The generation of random binary unordered trees"
    # [(0, -0.877), (1, -0.168)]    # "The intersection graph of paths in trees"
    # [(0, -0.910), (1, -0.141)]    # "Graph minors IV Widths of trees and well quasi ordering"
    # [(0, -0.617), (1, 0.054)]     # "Graph minors A survey"
    lsi.save(work_path + os.sep + 'model.lsi')              # same for tfidf, lda, ...
    lsi = models.LsiModel.load(work_path + os.sep + 'model.lsi')


def l4_Experiments():
    # 假设我有wiki的数据
    # load id->word mapping (the dictionary), one of the results of step 2 above
    id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
    # load corpus iterator
    mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output (recommended)
    # -------------------------- 1、Latent Semantic Analysis --------------------------
    # extract 400 LSI topics; use the default one-pass algorithm
    lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)
    lsi.print_topics(10)
    # -------------------------- 2、Latent Dirichlet Allocation --------------------------
    # extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
    # print the most contributing words for 20 randomly selected topics
    lda.print_topics(20)
    # extract 100 LDA topics, using 20 full passes, no online updates
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)
    # doc_lda = lda[doc_bow]
    pass




# ============================ 配置文件 ============================
work_path = r'F:\test_gensim'


if __name__ == '__main__':
    pass




import pandas as pd
import numpy as np
b = [[x, str(x+np.random.rand())] for x in range(103)]
a = pd.DataFrame(b, columns=['a','b'])
a['b'] = map(lambda x: 'aa'+ str(x[1]) if x[0] > 10 else str(x[1]), a.loc[:,['a','b']].values)
# 代码的意思是：对于dataframe：a，当'a'列大于10的时候，'b'列增加前缀'aa'。


