# -*- coding: utf-8 -*-
import os
import jieba
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.externals import joblib  # 二进制模型
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec


# 载入数据，预处理，切分训练集与测试集
def load_file_and_preprocessing():
    pos = pd.read_excel('dataset/pos.xls', header=None, index=None)
    neg = pd.read_excel('dataset/neg.xls', header=None, index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('./models/y_train.npy', y_train)
    np.save('./models/y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量求均值，来生成一个句子的vector
def sentence2vector(text, size, w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_vectors(x_train, x_test):
    dimensions = 300
    # 初始化模型和词表
    w2v = Word2Vec(x_train, size=dimensions, min_count=10)
    # 在评论训练集上建模
    # w2v.build_vocab(x_train)
    # w2v.train(x_train)

    train_vectors = np.concatenate([sentence2vector(z, dimensions, w2v)for z in x_train])
    np.save('./models/train_vectors.npy', train_vectors)
    print(train_vectors.shape)

    w2v.train(x_test, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    w2v.save('./models/w2v_model.pkl')

    test_vectors = np.concatenate([sentence2vector(z, dimensions, w2v) for z in x_test])
    np.save('./models/test_vectors.npy', test_vectors)
    print(test_vectors.shape)


def get_data():
    train_vectors = np.load('./models/train_vectors.npy')
    y_train = np.load('./models/y_train.npy')
    test_vectors = np.load('./models/test_vectors.npy')
    y_test = np.load('./models/y_test.npy')
    return train_vectors, y_train, test_vectors, y_test


# 训练SVM模型
def train_svm(train_vectors, y_train, test_vectors, y_test):
    classifier = SVC(kernel='rbf', verbose=True)
    classifier.fit(train_vectors, y_train)
    joblib.dump(classifier, './models/chinese_sentiment_analysis_svm.pkl')
    print(classifier.score(test_vectors, y_test))


# 构建待预测句子的向量
def get_predict_vectors(words):
    dimensions = 300
    w2v = Word2Vec.load('models/w2v_model.pkl')
    vectors = sentence2vector(words, dimensions, w2v)
    return vectors


# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    vectors = get_predict_vectors(words)
    classifier = joblib.load('./models/chinese_sentiment_analysis_svm.pkl')
    result = classifier.predict(vectors)
    if result[0] == 1:
        print('positive')
    else:
        print('negative')

if __name__ == '__main__':
    if not os.path.exists('./models/chinese_sentiment_analysis_svm.pkl'):
        print('Step 1: Loading And Preprocessing Data ...')
        x_train, x_test = load_file_and_preprocessing()

        print('Step 2: Computing Word2Vec ...')
        get_vectors(x_train, x_test)

        print('Step 3: Getting Prepared Model Input ...')
        train_vectors, y_train, test_vectors, y_test = get_data()

        print('Step 4: Training SVM Model ...')
        train_svm(train_vectors, y_train, test_vectors, y_test)

    print('Loading Model ...')
    # 带预测情感的句子
    string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    svm_predict(string)
    string = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string)
