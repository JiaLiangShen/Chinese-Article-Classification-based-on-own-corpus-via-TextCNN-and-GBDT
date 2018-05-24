# Article-Classification-based-on-own-corpus-via-TextCNN-and-GBDT
## Create Time: 2018-05-24
本文从处理个人语料库（可替换成你自己的）开始，搭建了两套最简易的基础模型来做文本的分类问题  
Data Source里为自己手动整理的简易语料库，toy级玩具，直接下载后跑模型文件即可。  

1、统计语料库，文本的的TF-IDF值作为词向量 + GBDT(Gradient Boosting Classifier)/SVM(Support Vector Classifer)  
2、语料库的Word2Vector作为词向量 + 标准单层TextCNN(Keras Version)  
主要第三方依赖包：sklearn, gensim, jieba,pandas,scipy,Keras，Tensorflow  

langcov.py中文简繁体转换,zh_wiki.py(大陆语/台湾语，繁简体转换词典),感谢skydark@github 源文件连接：  
https://raw.githubusercontent.com/skydark/nstools/master/zhtools/langconv.py  
https://raw.githubusercontent.com/skydark/nstools/master/zhtools/zh_wiki.py  

## 文件说明：  
1、Articles_Classifer_TF-IDF_GBDT.py 文件为决策树模型及TF-IDF值为词向量  
2、Articles_Classifier_Word2Vector_TextCNN_Keras.py 为最简易的卷积神经网络的分类模型  
架构：1D-Convolutional Layer + 1-Max Pooling Layer + Flatten Layer+ Dropout Layer+ Dense Layer + Softmax Layer  
3、Extract_Articles_zhwiki_latest_pages.py 抽取最新的Wiki中文语料库的文件，路径换成自己的即可，将Json格式转换为一行一篇文章存储在txt中  
4、Word2Vector_Training.py 训练自己/Wiki_Zh的词向量，可以根据自己需求更改，Comment掉的Code可以根据需求切换。
