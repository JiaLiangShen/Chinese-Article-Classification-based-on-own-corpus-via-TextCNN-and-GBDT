"""
从Wikipedia_zh 将JSON的格式里的文章提取出来。
1、存入序列化文件wiki_zh_vocal_200K.mm（Matrix Market格式）对应为Bag of Word词向量形式，得到.mm与.index.mm文件
2、直接写入txt文件中，txt庞大，内存大可以一次性读入，可视化好，直接打开txt文本阅读正文

问题：
在处理中文简体和中文繁体的的时候，使用目标编码中不存在的中文字符，也会导致UnicodeEncodeError
LogInfo:
Traceback (most recent call last):
  File "E:/Working_Code/Information_Database/Extract_Articles_zhwiki_latest_pages.py", line 39, in <module>
    output.write(space.join(text) + "\n")
UnicodeEncodeError: 'gbk' codec can't encode character '\xf6' in position 892: illegal multibyte sequence
解决办法：
先转简繁体，安装/调用OpenCC失败，放弃

最大原因：windows底层编码，从入门python到经历编码问题后放弃系列
"""
# -*- coding: utf-8 -*-
# # ! /usr/bin/env python
import os
from gensim.corpora import WikiCorpus
import logging
import warnings
import gensim
import datetime
# import sys
# import io
import codecs
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.setLevel(level=logging.INFO)
gensim.corpora.wikicorpus.ARTICLE_MIN_WORDS = 50
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # 改变标准输出的默认编码，貌似没什么用


def extract_store_txt_wiki_zh():
    # Training Articles Path:
    # E:\Working_Code\word embedding\word2vec_training\zhwiki-latest-pages-articles.xml\zhwiki-latest-pages-articles.xml
    start_time = datetime.datetime.now()
    path_main = 'E:/Working_Code/word embedding/word2vec_training/'
    input_file = os.path.join(path_main, "zhwiki-latest-pages-articles.xml.bz2")
    wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})

    end_time = datetime.datetime.now()
    time_cost = (end_time - start_time).seconds

    space = ' '
    i = 0
    output = codecs.open('extracted_data.txt', 'w', encoding='utf-8')
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if i % 10000 == 0:
            print("Saved " + str(i) + " articles")
    output.close()
    print("Finished Saved " + str(i) + " articles")
    print(time_cost)

    # # 序列化词向量，可选项
    # MmCorpus.serialize('wiki_zh_vocab200k.mm', wiki)
    # corpus = MmCorpus('wiki_zh_vocab200k.mm')


if __name__ == '__main__':
    extract_store_txt_wiki_zh()
