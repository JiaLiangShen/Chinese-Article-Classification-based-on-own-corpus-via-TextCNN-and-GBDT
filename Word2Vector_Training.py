"""
训练中文的词向量，去除停用词，数字，英文
由于编码问题解决不了，直接训练自己的语料库的word2vector
"""
# -*- coding: utf-8 -*-
# ! /usr/bin/env python
import logging
import os
import codecs
import jieba
import multiprocessing
import sys
import re
import pandas as pd
from gensim.models import Word2Vec
import warnings
from gensim.models import word2vec
from langconv import Converter
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.setLevel(level=logging.INFO)


# 繁体字转简体
def traditional_to_simplified(sentence):
    # 将sentence中的繁体字转为简体字
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


# 处理未加工过的文档
def process_raw_articles(path, main_articles):
    processed_articles = []
    stopwords_main = load_stopwords(path)

    # 分词 + 过滤停用词表
    for article in main_articles:
        seg_ed_article = []
        article = re.sub("[A-Za-z0-9]", "", article)  # 正则表达式，去掉所有数字和英文
        seg_list = jieba.cut(article)
        for word in seg_list:
            if word not in stopwords_main:
                seg_ed_article.append(word)
        processed_articles.append(seg_ed_article)
    return processed_articles


# 读停用词表
def load_stopwords(path=sys.path[0]):
    stopwords = codecs.open(os.path.join(path, 'stopwords.txt'), 'r', 'utf-8').read()
    return stopwords


if __name__ == '__main__':

    '''
    词向量训练，这部分为自己的语料库训练
    '''
    # path_main = sys.path[0] + '/Data Source'
    # data = pd.read_csv(os.path.join(path_main, 'corpus_articles.csv'), encoding='gbk')
    # processed_data = process_raw_articles(path_main, data.articles.values)
    # print(processed_data)

    # i = 0
    # space = ' '
    # current_file = codecs.open('processed_articles_for_w2v_train.txt', 'w', encoding='utf-8')
    # for text in processed_data:
    #     current_file.write(space.join(text) + "\n")
    #     i = i + 1
    #     if i % 100 == 0:
    #         print("Saved " + str(i) + " articles")
    # current_file.close()

    # sentences = word2vec.Text8Corpus("processed_articles_for_w2v_train.txt")
    #
    # model = Word2Vec(sentences, size=200, window=5, min_count=1,
    #                  workers=multiprocessing.cpu_count(), sample=0.001, sorted_vocab=True)
    # model.wv.save_word2vec_format('word2vector_hr_info.bin', binary=False)

    '''
    词向量训练，这部分为针对中文维基百科的训练
    '''
    #  For Wiki_Zh_Corpus处理繁体转变为简体，这里开始是处理wiki的文件
    # f_read = open(os.path.join(sys.path[0], 'extracted_data.txt'), 'r', encoding='utf-8')
    # f_write = open(os.path.join(sys.path[0], 'processed_extracted_data.txt'), 'w', encoding='utf-8')
    # for line in f_read:
    #     sentence = Converter('zh-hans').convert(line)
    #     f_write.write(sentence)
    # f_read.close()
    # f_write.close()

    # path_main = sys.path[0] + '/Data Source'
    # data = codecs.open('processed_extracted_wiki_data.txt', 'r', encoding='utf-8')
    # processed_data = process_raw_articles(path_main, data)
    #
    # i = 0
    # space = ' '
    # current_file = codecs.open('clear_wiki_zh_data.txt', 'w', encoding='utf-8')
    # for text in processed_data:
    #     current_file.write(space.join(text) + "\n")
    #     i = i + 1
    #     if i % 100 == 0:
    #         print("Saved " + str(i) + " articles")
    # current_file.close()

    sentences = word2vec.Text8Corpus("clear_wiki_zh_data.txt")

    model = Word2Vec(sentences, size=200, window=5, min_count=10,
                     workers=multiprocessing.cpu_count(), sample=0.001, sorted_vocab=True)
    model.wv.save_word2vec_format('word2vector_wiki_zh_info.bin2', binary=False)


