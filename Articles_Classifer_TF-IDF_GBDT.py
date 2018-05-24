"""
Author: AceKimi
对未标识标签的文章分类
Steps:
1、对训练文本进行归一化、停用词、去低频次（语料库过少，为了保留Feature，未作）
及特征工程（务必将训练文本放入语料库，不然字典长度及单词缺失将会非常影响分类性能）
2、将所有训练文档表达为未Feature：tf-idf值的词向量（需要调用Scipy的密度向量函数）
3、扔进SVM和GBDT的分类模型去跑


Input Feature: TF-IDF, Word Embedding(Word2Vector, not large enough domain corpus)
Classifier: Support Vector Machine,Gradient Boosting Decision Trees
Optimizer: GridSearch & SGD, depends on all situations(Not Implement)
Output:Label
"""

# -*- coding: utf-8 -*-
# ! /usr/bin/env python
import datetime
import sys
import pandas as pd
from gensim import corpora
import codecs
import re
import os
from gensim import models
import logging
import jieba
import warnings
import numpy as np
import scipy.sparse.csr as csr
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from textrank4zh import TextRank4Keyword, TextRank4Sentence
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 对tf-idf值做数据平滑处理
def reload_tf2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    if docfreq == totaldocs:
        return 1 + np.log(float(totaldocs) / docfreq) / np.log(log_base)
    else:
        return add + np.log(float(totaldocs) / docfreq) / np.log(log_base)


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


# 给出文本的tf-idf值格式
def tf_idf_representation(dictionary, raw_txt):
    vec_bow = [dictionary.doc2bow(text, return_missing=False) for text in raw_txt]
    tf_idf_model = models.TfidfModel(vec_bow, dictionary=dictionary, normalize=True, wglobal=reload_tf2idf)
    # 对于没做数据平滑的Gensim，修改公式,分子+1,相当于重载底层函数
    article_tf_id = tf_idf_model[vec_bow]
    return article_tf_id


# 返回字典长度
def get_dictionary_length(dictionary):
    return len(dictionary)


# 以字典长度为标准，补全所有文章的tf-idf向量，空缺的话为0，处理成SVM可读的密集向量
def sparse2dense_vector(corpus, dictionary):
    training_data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            training_data.append(elem[1])
        line_count += 1
    tf_idf_sparse_matrix = csr.csr_matrix((training_data, (rows, cols)), shape=[max(rows)+1, len(dictionary)])  # 稀疏向量
    tf_idf_matrix = tf_idf_sparse_matrix.toarray()  # 密集向量
    return tf_idf_matrix


# 对标签做编码
def label_encoder(article_labels):
    labels_encoder = LabelEncoder()
    labels_encoder.fit_transform(article_labels)
    return labels_encoder.transform(article_labels), labels_encoder


# SVM做文章分类,资料太少,完全underfit
def classify_svm(data_sets, label_sets):
    params = {'kernel': 'rbf', 'verbose': 0, 'C': 1.0, 'decision_function_shape': 'ovr', 'max_iter': -1}
    clf = SVC(**params)
    clf.fit(data_sets, label_sets)
    # print(clf.score(data_sets, label_sets))
    return clf


# 用集成决策树做文章分类,容易overfit
def classify_gbc(data_sets, label_sets):
    params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'deviance', 'verbose': 0}
    clf = GradientBoostingClassifier(**params)
    clf.fit(data_sets, label_sets)
    # print(clf.score(data_sets, label_sets))
    return clf


# 通过语料库生成词典，返回词典，未保存
def make_dictionary(path=sys.path[0]):
    data = pd.read_csv(os.path.join(path, 'corpus_articles.csv'), encoding='gbk')
    main_articles = data.articles.values
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

    # 生成字典和向量语料
    dictionary = corpora.Dictionary(processed_articles)
    dictionary.filter_extremes(no_below=5)
    return dictionary


# Training Model
# 传入训练的语料库和模型类型（默认GBDT classifier）
def training_classify_model(path, model_class='gbc'):
    data = pd.read_csv(os.path.join(path, 'training_articles.csv'), encoding='gbk')
    articles = data.articles.values
    labels = data.label.values
    dictionary = make_dictionary(path)
    processed_articles_main = process_raw_articles(path, articles)
    article_tf_idf = tf_idf_representation(dictionary, processed_articles_main)

    tf_idf_matrix_main = sparse2dense_vector(article_tf_idf, dictionary)
    transformed_label, label_decoder = label_encoder(labels)
    if model_class == 'svc':
        return classify_svm(tf_idf_matrix_main, transformed_label), dictionary, label_decoder
    elif model_class == 'gbc':
        return classify_gbc(tf_idf_matrix_main, transformed_label), dictionary, label_decoder


# 抽取关键字，num_key_word为个数
def key_word_extraction(articles):
    key_word = []
    num_key_word = 5
    tr4w = TextRank4Keyword()
    for i in range(len(articles)):
        temp_key_word = []
        tr4w.analyze(text=articles[i], lower=True, window=3, vertex_source='all_filters', edge_source='all_filters')
        for j in range(num_key_word):
            temp_key_word.append(tr4w.get_keywords(num=num_key_word)[j]['word'])
        key_word.append(temp_key_word)
    return key_word


# 返回List文章的摘要，num_sentence为摘要句子数
def abstract_extraction(articles):
    abstract_sentence = []
    num_sentence = 1
    tr4w = TextRank4Sentence()
    for i in range(len(articles)):
        tr4w.analyze(text=articles[i])
        abstract_sentence.append(tr4w.get_key_sentences(num=num_sentence)[0]['sentence'])
    return abstract_sentence


# 得到需要各类文章的摘要及关键词
def get_keyword_abstract(data):
    key_word = key_word_extraction(data)
    key_sentence = abstract_extraction(data)
    return key_word, key_sentence


# 处理原始文档
def process_origin_articles(path, articles, dictionary):
    processed_articles = process_raw_articles(path, articles)
    word_vector = tf_idf_representation(dictionary, processed_articles)
    dense_vector = sparse2dense_vector(word_vector, dictionary)
    return dense_vector

# 所有处理的文件请在这里修改，主函数
def article_info_extract_classify():
    start_time = datetime.datetime.now()
    path_data_source_main = os.path.join(sys.path[0], 'Data Source')  # 主要路径，字典，训练样本等存放地
    path_result_main = os.path.join(sys.path[0], 'Result')
    raw_data_path = sys.path[0]  # 未处理的样本路径
    raw_data = pd.read_csv(os.path.join(raw_data_path, 'test_data.csv'), encoding='gbk')
    raw_data_articles = raw_data.articles.values

    # 请务必在当前系统目录下，建立Data Source文件夹（所有预训练/预处理的文件都应该放在path_main变量下）
    # 指定文件名的3个文件：（然后调用trained_model就可以得到模型分类器，默认为GBDT）
    # 1、stopwords.txt 停用词表
    # 2、corpus_articles.csv 语料库
    # 3、training_samples.csv 训练库
    # 需要进行分类、关键词、文章摘要的文本可以指定某个特殊位置
    # 1、test_data.csv 需要做分类、关键词、文章摘要的未加工文本
    trained_model, dictionary_main, decoder = training_classify_model(path_data_source_main, model_class='gbc')
    raw_data_main = process_origin_articles(path_data_source_main, raw_data_articles, dictionary_main)
    labels_main = trained_model.predict(raw_data_main)
    labels_main = decoder.inverse_transform(labels_main)
    key_word_main, abstract_main = get_keyword_abstract(raw_data_articles)

    end_time = datetime.datetime.now()
    time_cost = (end_time - start_time).seconds
    final_data = pd.DataFrame({'articles': raw_data_articles, 'label': labels_main,
                               'key_word': key_word_main, 'abstract': abstract_main})
    # Output标注完的文件在当前文件夹下，可以修改
    final_data.to_csv(os.path.join(path_result_main, 'final_data.csv'), index=True, encoding='gbk')
    # 打印所需时间
    print(time_cost)


if __name__ == "__main__":
    article_info_extract_classify()
