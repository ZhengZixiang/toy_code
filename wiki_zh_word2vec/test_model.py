# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('/data/wiki.zh.text.model')

result = model.most_similar(u'足球')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'男人')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'女人')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'青蛙')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'姨夫')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'衣服')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'公安局')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'铁道部')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'清华大学')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'卫视')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'林丹')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'语言学')
for e in result:
    print(e[0], e[1])
print('--------------------------------')
result = model.most_similar(u'计算机')
for e in result:
    print(e[0], e[1])
print('--------------------------------')

print(model.similarity(u'计算机', u'自动化'))
print(model.similarity(u'女人', u'男人'))
print(model.doesnt_match(u'早餐 晚餐 午餐 中心'.split()))
