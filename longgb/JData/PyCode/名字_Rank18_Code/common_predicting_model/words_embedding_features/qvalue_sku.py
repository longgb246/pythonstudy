#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    q_words = {}
    q_feature_lens = -1
    for line in open("common_predicting_model/words_embedding_features/input/wordsQvalue.txt"):
        segs = line.strip('\n').split('\t')
        if float(segs[1]) > 0:
            q_feature_lens += 1
            q_words[segs[0]] = (q_feature_lens, float(segs[1]))

    name = {}
    for line in open("common_predicting_model/words_embedding_features/input/embedding.rawdata"):
        segs = line.strip('\n').split('\t')
        name[segs[0]] = segs[2].split("")

    emb = {}
    for line in open("common_predicting_model/words_embedding_features/input/lisai"):
        segs = line.strip('\n').split('\t')
        emb[segs[0]] = segs[1].split(',')

    print "start to joint train data set..."
    f = open("common_predicting_model/words_embedding_features/output/f1_trainset.data", "w")
    for line in open("common_predicting_model/basic_model_features/output/trainset.data"):
        segs = line.strip('\n').split('\t')
        delta = (q_feature_lens + 1) * [0]
        if segs[0] in name:
            for word in name[segs[0]]:
                if word in q_words:
                    delta[q_words[word][0]] = q_words[word][1]

        f.write(line.strip('\n') + '\t' + '\t'.join(map(lambda x: str(x), delta)) + '\t' + '\t'.join(emb[segs[0]]) + '\n')
    f.close()

    print "start to joint test data set..."
    f = open("common_predicting_model/words_embedding_features/output/f1_predictset.data", "w")
    for line in open("common_predicting_model/basic_model_features/output/predictset.data"):
        segs = line.strip('\n').split('\t')
        delta = (q_feature_lens + 1) * [0]
        if segs[0] in name:
            for word in name[segs[0]]:
                if word in q_words:
                    delta[q_words[word][0]] = q_words[word][1]
        f.write(line.strip('\n') + '\t' + '\t'.join(map(lambda x: str(x), delta)) + '\t' + '\t'.join(emb[segs[0]]) + '\n')
    f.close()

