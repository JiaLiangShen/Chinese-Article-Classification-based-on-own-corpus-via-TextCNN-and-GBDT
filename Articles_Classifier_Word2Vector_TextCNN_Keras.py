"""
依然是用Word Embedding作Feature，只不过把每篇文章的tf-idf词向量值换为Word2Vector值，pre-train使用人力资源语料库
设置卷积神经网络架构，1层C-P，1层Dense，输出层 n_class、softmax
Model:TextCNN, 1 layer 1D Convolutional,1-max(Try K-max以后再说) pooling + 1 layer dense + softmax
Create Time: 2018-05-17
AceKimi@ZY.bigdata

Update:
2018-05-14
原：Word2Vector是使用交大沈老师合作项目学生所提供的（现有语料库太少，训练没任何意义）
Training出来无法使用(vectors.bin)，没有过滤数字/英文，没有用停用词表。
现：重新Training，Based on人力资源语料库（自己收集）
Keras版本

2018-05-17
1、将文本处理成标准格式：一篇文章一行，分词。将文章转换成word2vector形式。
2、定义数据流，处理成Tensor形式。
3、定义网络架构

2018-05-18
1、将文本格式处理完整，包括截团/补零，处理为Tensor并保存
2、设计了标准TextCNN结构
KeyedVectors Refer: https://radimrehurek.com/gensim/models/deprecated/keyedvectors.html
Paper Based On <Convolutional Neural Network for Sentence Classification> by YoonKim

2018-05-21
1、调整了未登录词w2v的分布的概率密度函数
2、跑通
3、最终loss为softmax (One-Hot-Encoding)

"""
import pandas as pd
import os
import sys
import logging
import numpy as np
import codecs
import jieba
import re
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, InputLayer
# from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
sys.path.append(sys.path[0])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
WORD_2_VECTOR_DIM = 200  # w2c的维数
SENTENCE_MAX_LENGTH = 500  # 句子最多可包含的单词数
np.random.seed(np.random.randint(99999))


def label_encoder_func(article_labels):
    labels_encoder = LabelEncoder()
    a = labels_encoder.fit_transform(article_labels)
    a = a.reshape(-1, 1)
    labels_encoder_one_hot = OneHotEncoder(sparse=True)
    transformed_label_one_hot = labels_encoder_one_hot.fit_transform(a)
    return transformed_label_one_hot, labels_encoder, labels_encoder_one_hot


def load_stopwords(path='./Data Source'):
    stopwords = codecs.open(os.path.join(path, 'stopwords.txt'), 'r', 'utf-8').read()
    return stopwords


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


def raw_article2vector(articles, path='./Data Source'):
    processed_data = process_raw_articles(path, articles)
    model = KeyedVectors.load_word2vec_format(
        os.path.join(path, 'word2vector_hr_info.bin'), binary=False)
    data_2_wv = []
    zero_vectors = np.zeros(WORD_2_VECTOR_DIM)
    # 截断和补全功能可以用以下函数代替，但是具体跑出来的结果是全为零，有误，今后再尝试
    # 这里采取小于500时直接在末尾补零。
    # 字典里没有的词的w2v直接用 np.random.uniform(-0.25, 0.25, WORD_2_VECTOR_DIM)代替
    # x_train = sequence.pad_sequences\
    #     (data_2_wv, maxlen=SENTENCE_MAX_LENGTH, padding='post', truncating='post')

    for document in processed_data:
        temp_list = []
        if len(document) >= SENTENCE_MAX_LENGTH:
            for i in range(SENTENCE_MAX_LENGTH):
                if document[i] in model:
                    temp_list.append(model[document[i]])
                else:
                    temp_list.append(np.random.uniform(-0.25, 0.25, WORD_2_VECTOR_DIM))
        else:
            for i in range(len(document)):
                if document[i] in model:
                    temp_list.append(model[document[i]])
                else:
                    temp_list.append(np.random.uniform(-0.25, 0.25, WORD_2_VECTOR_DIM))
            temp_list.extend([zero_vectors for j in range(SENTENCE_MAX_LENGTH - len(document))])
        data_2_wv.append(temp_list)
    return data_2_wv


def process_training_data(path='./Data Source'):
    data = pd.read_csv(os.path.join(path, 'training_articles.csv'), encoding='gbk')
    x_train = raw_article2vector(data.articles.values)
    label_train_data = data.label.values
    x_label, x_label_encoder, x_label_encoder_one_hot = label_encoder_func(label_train_data)
    return x_train, x_label


# 1、根据原始paper，有几种对pre-training的处理方式，这里只实现了static_CNN，即训练出的w2v不会被更新
# 相当于 Weight_matrix * b , b = [1,1,1,1,1,1,1], len(b) = 200
# 2、 Non-static模式，即在pre-train到Convolutional层再接一层全连接层学习Weight_matrix的二次映射，日后实现
def textCNN_training(training_data, training_label, model_type='static_embedding_weight'):
    # Model Hyperparameters
    filter_sizes = 3  # 相当于BOW的Window_Size
    num_filters = 10
    dropout_prob = 0.5
    hidden_dims = 100
    batch_size = 16
    num_epochs = 100
    num_class = 3

    if model_type == 'static_embedding_weight':
        print('Static_Word_Embedding_Training')
        model = Sequential()
        model.add(InputLayer(input_shape=(SENTENCE_MAX_LENGTH, WORD_2_VECTOR_DIM)))
        model.add(Convolution1D(filters=num_filters,
                                kernel_size=filter_sizes,
                                padding="valid",
                                activation="relu",
                                strides=1,))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())  # 压平，这里是单通道，事实上原始paper是双通道
        model.add(Dropout(rate=dropout_prob))
        model.add(Dense(hidden_dims, activation="relu"))
        model.add(Dense(num_class, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(training_data, training_label, epochs=num_epochs,  verbose=2,
                  batch_size=batch_size, validation_data=(training_data[130:], training_label[130:]),)
        return model

    #  使用的Word2Vector不同，针对不同语料库Fine Tuning过的Word2Vector，架构是一样的
    elif model_type == 'non_static_embedding_weight':
        print('Non_Static_Word_Embedding_Training')
    # 根据语料库进行One-Hot Encoding
    elif model_type == 'rand_embedding_weight':
        print('Random_Word_Embedding_Weight, Test for Model BaseLine')


if __name__ == '__main__':
    # Training CNN
    training_data_main, training_label_main = process_training_data()
    x_train_main = np.stack([np.stack([word for word in sentence]) for sentence in training_data_main])  # 变成Tensor
    model_cnn = textCNN_training(x_train_main, training_label_main)

    # Read Testing Data
    path_main = os.path.join(sys.path[0], 'Data Source')
    raw_data = pd.read_csv(os.path.join(sys.path[0], 'test_data.csv'), encoding='gbk')
    raw_data_articles = raw_data.articles.values  # 文本
    test_articles_2_wv = raw_article2vector(raw_data_articles, path_main)  # 将所有文章分词及处理成词向量
    x_test_main = np.stack([np.stack([word for word in sentence]) for sentence in test_articles_2_wv])
    print(model_cnn.predict_classes(x_test_main, verbose=2))



