import jieba.posseg as pseg
import jieba
import codecs
from gensim import corpora
from gensim.summarization import bm25
import os
import re
import sys
from pyhanlp import HanLP

'''
参数：
第一个 训练文件 domain query 形式
第二个 测试文件 domain query 形式

'''

def bm25_model(pattern):      # 'jieba' or 'hanlp'
    corpus = []
    filenames = []
    file = sys.argv[1]
    file_test = sys.argv[2]

    def hanlp_seg(query):
        words =[]
        for term in HanLP.segment(query):
            words.append(term.word)
        return words

    def jieba_seg(query):
        return jieba.lcut_for_search(query)

    def seg(query):
        if pattern == 'jieba':
            seg = jieba_seg(query)
            return seg
        elif pattern == 'hanlp':
            seg = hanlp_seg(query)
            return seg
        else:
            print('choose one seg pattern: hanlp or jieba')


    with open(file,'r') as f:
        for line in f.readlines():
            domain,query = line.strip().split()



            # print(domain,query)

            query_array = seg(query)
            # print(query_array)

            corpus.append(query_array)
            filenames.append(domain)


    dictionary = corpora.Dictionary(corpus)
    # print( len(dictionary))


    #用gensim建立BM25模型
    bm25Model = bm25.BM25(corpus)
    #根据gensim源码，计算平均逆文档频率
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

    def test_query(query_str,filenames,corpus):
        # query_str = '2g网络去掉'
        query_str = seg(query_str)

        query = []

        for word in query_str:
            query.append(word)
        scores = bm25Model.get_scores(query,average_idf)
        # scores.sort(reverse=True)
        # print scores


        idx = scores.index(max(scores))

        fname = filenames[idx]

        # print(fname,corpus[idx])
        return fname

    with open(file_test,'r') as f:
        sum_query = 0
        predict_right = 0
        for line in f.readlines():

            golden_domain,query = line.strip().split()

            predict_domain = test_query(query,filenames,corpus)

            sum_query += 1
            if golden_domain == predict_domain:
                predict_right +=1
            # else:
                # print()

                # print(query,golden_domain,predict_domain)

    print(str(pattern) +' accuracy:',predict_right/sum_query)


# you can choose two pattern : 'jieba' or 'hanlp'

bm25_model('jieba')
bm25_model('hanlp')


