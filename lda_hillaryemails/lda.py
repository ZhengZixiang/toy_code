import numpy as np
import pandas as pd
import re
import gensim
from gensim import corpora, models, similarities

# LDA模型应用：一眼看穿希拉里的邮件
df = pd.read_csv('./HillaryEmails.csv')
df = df[['Id', 'ExtractedBodyText']].dropna()


# 文本预处理
def clean_email_text(text):
    text = text.replace('\n', ' ')  # 新行是我们不需要的
    text = re.sub(r'-', ' ', text)  # 把’-’的两个单词分开
    text = re.sub(r'\d+/\d+/\d+', '', text)  # 日期，对主题模型没意义
    text = re.sub(r'[\w]+@[\.\w]+', '', text)  # 时间，对主题模型没意义
    text = re.sub(r'/[a-zA-Z]*[:\//\]*[a-zA-Z0-9\-_]+\.+[a-zA-Z0-9\.\/%&=\?\-_]+/i', '', text)  #网址，对主题模型没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词直接排除，就只剩下有意义的单词，例如don't用don表示
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


# 现在新建一个column并把我们的预处理方法跑一遍
docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))

print('第一列预处理结果：', docs.head(1).values)

# 我们直接把所有的邮件内容拿出来
doclist = docs.values

# 用gensim进行LDA模型构建
# 我们把doclist的内容转化成分词结果

# 人工分词
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just',
            'her', 'ours', 'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such',
            'too', 'mustn', 'under', 'their', 'if', 'to', 'my', 'himself', 'after', 'why',
            'while', 'can', 'each', 'itself', 'his', 'all', 'once', 'herself', 'more', 'our',
            'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 'didn',
            'nor', 'as', 'now', 'before', 'yours', 'those', 'from', 'who', 'was', 'm', 'been',
            'will', 'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't',
            'mightn', 'she', 'again', 'be', 'by', 'shan', 'have', 'yourselves', 'needn', 'and',
            'are', 'o', 'these', 'further', 'most', 'yourself', 'having', 'aren', 'here', 'he',
            'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 'when',
            'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom',
            'wouldn', 'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until',
            'won', 'no', 'about', 'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing',
            'an', 'or', 'ain', 'hers', 'wasn', 'weren', 'above', 'a', 'at', 'your', 'theirs', 'below',
            'other', 'not', 're', 'him', 'during', 'which']
texts = [[word for word in doc.lower().split() if word not in stoplist]for doc in doclist]

# 查看是否分词成功
print('第一列分词结果：', texts[0])

# 建立语料库，用标记化方法，把每个单词用一个数字index指代，把原文变成数组
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# 其中每对数字第一个数字表示单词index，第二个数字表示在这封邮件中这个单词出现了几次
print('第一列doc2bow结果：', corpus[0])

# 建模
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 打印出第10个主题的top5重要的单词
print('主题测试：', lda.print_topic(10, 5))

# 打印出所有主题的top5重要的单词
print('所有主题：', lda.print_topics(20, 5))

# 接下来可以通过下面两个方法把新鲜的文本或单词分成20个主题中的一个
# lda.get_document_topics(bow)
# lda.get_term_topics(word_id)
with open('./test.txt', 'r') as f:
    i = 1
    for line in f.readlines():
        if line.strip().strip('\n') != '':
            print('第{num}句'.format(num=i))
            i += 1

            cleaned = clean_email_text(line)
            words = [word for word in cleaned.lower().split() if word not in stoplist]
            # print(dictionary.doc2bow(words))
            print(lda.get_document_topics(dictionary.doc2bow(words)))
