#-*- coding:utf-8 -*-
import os
import codecs
import jieba
import warnings
warnings.filterwarnings('ignore')
from gensim import corpora, models, similarities
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def getSourceFileLists(sourceDataDir):
    '''
    获取数据的list
    '''
    subDirList = os.listdir(sourceDataDir)
    fileLists = [x for x in subDirList if os.path.isfile(sourceDataDir +os.sep + x)]
    return fileLists


def mkdir(path):
    '''
    创建文件夹
    '''
    if not os.path.exists(path):
        os.mkdir(path)



# 数据源目录
rootDataDir = r'D:\Lgb\NewSpace\pythonstudy\longgb\life\School\Graduate\Longgb246\TextMining\txthj_264'
sourceDataDir = rootDataDir + os.sep + 'data'



# 数据源文件列表
fileLists = getSourceFileLists(sourceDataDir)

if 0 < len(fileLists):

    punctuations = ['', '\n', '\t', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

    mkdir(rootDataDir + os.sep + 'dict')
    mkdir(rootDataDir + os.sep + 'corpus')

    import chardet

    for fileName in fileLists:
        chardet.detect(fileName)
        print fileName
        print fileName.decode('gb2312')
        print type(fileName)
        break
        hFile = None
        content = None
        try:
            hFile = codecs.open(fileName, 'r', 'gb18030')
            content = hFile.readlines()
        except Exception, e:
            print e
        finally:
            if hFile:
                hFile.close()
        if content:
            fileFenci = [x for x in jieba.cut(' '.join(content), cut_all=True)]
            fileFenci2 = [word for word in fileFenci if not word in punctuations]
            texts = [fileFenci2]
            all_tokens = sum(texts, [])
            tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
            texts = [[word for word in text if word not in tokens_once] for text in texts]
            sFileDir, sFileName = os.path.split(fileName)
            dictFileName = 'dict/' + sFileName + '.dict'
            corpusFileName = 'corpus/' + sFileName + '.mm'
            dictionary = corpora.Dictionary(texts)
            dictionary.save_as_text(dictFileName)
            corpus = ([dictionary.doc2bow(text) for text in texts])
            corpora.MmCorpus.serialize(corpusFileName, corpus)


print 'Build corpus done'
