# -*- coding: utf-8 -*-
import nltk
import sys
from nltk.corpus import brown

# import nltk
# nltk.download('brown')

# 预处理词库，给词加上开始和结束符，（START，START）（END，END）
brown_tags_words = []
for sent in brown.tagged_sents():
    # 先加开头
    brown_tags_words.append(('START', 'START'))
    # 为了省事，我们把tag都省略成前两个字母
    brown_tags_words.extend([(tag[:2], word) for (word, tag) in sent])
    # 最后加结束
    brown_tags_words.append(('END', 'END'))

# 词统计
# B: P(wi | ti) = count(wi, ti) / count(ti)
# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

# 查看统计结果
print('The probability of an adjective(JJ) being "new" is', cpd_tagwords['JJ'].prob('new'))
print('The probability of a verb(VB) being "duck" is', cpd_tagwords['VB'].prob("duck"))

# A: P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
# 先取出所有的tag
brown_tags = [tag for (tag, word) in brown_tags_words]
# count(t{i-1}, ti)
cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# P(t{i-1}, ti)
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

# 查看统计结果
print('If we have just seen "DT", the probability of "NN" is ', cpd_tags['DT'].prob('NN'))
print('If we have just seen "VB", the probability of "JJ" is ', cpd_tags['VB'].prob('JJ'))
print('If we have just seen "VB", the probability of "NN" is ', cpd_tags['VB'].prob('NN'))

# I want to race -> PP VB TO VB
prob_tagsequence = cpd_tags['START'].prob('PP') * cpd_tagwords['PP'].prob('I') * \
    cpd_tags['PP'].prob('VB') * cpd_tagwords['VB'].prob('want') * \
    cpd_tags['VB'].prob('TO') * cpd_tagwords['TO'].prob('to') * \
    cpd_tags['TO'].prob('VB') * cpd_tagwords['VB'].prob('race') * \
    cpd_tags['VB'].prob('END')
print('The probability of the tag sequence "START PP VB TO VB END" for "I want to race" is: ', prob_tagsequence)

# Viterbi实现
# 如果已经有一句话，我们怎么知道最符合的tag是哪组
# 首先取出所有独特的tags
distinct_tags = set(brown_tags)
# given sentence
sent = ['I', 'want', 'to', 'race']
sent_len = len(sent)

# 开始Viterbi，从1循环到句子长N，记为i，每次找出以tag x为最终节点，长度为i的链
viterbi = []
# 同时还需要一个backtracing回溯，把tag x前一个tag记下来
backpointer = []

first_viterbi = {}
first_backpointer = {}
for tag in distinct_tags:
    if tag == 'START': continue
    first_viterbi[tag] = cpd_tags['START'].prob(tag) * cpd_tagwords[tag].prob(sent[0])
    first_backpointer[tag] = 'START'
print(first_viterbi)
print(first_backpointer)

# 把上面的初始结果存到viterbi和backpointer两个变量里去
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

# 看一眼目前最好的tag是什么，max里的key是参数名，表示argmax，根据什么取最大，而不是first_viterbi的键
current_best = max(first_viterbi.keys(), key=lambda tag: first_viterbi[tag])
print('\nWord "' + sent[0] + '" current best two-tag sequence: ', first_backpointer[current_best], current_best)

# 以此类推开始loop
for index in range(1, sent_len):
    current_viterbi = {}
    current_backpointer = {}
    prev_viterbi = viterbi[-1]

    for tag in distinct_tags:
        if tag == 'START': continue
        # 如果现在这个tag是x，现在的单词是w
        # 我们想找前一个tag y，并且让最好的tag sequencey以 y x 结尾
        #　也就是说ｙ要能最大化
        # prev_viterbi[y] * P(x|y) * P(w|x)
        best_previous = max(prev_viterbi.keys(), key=lambda prev_tag: prev_viterbi[prev_tag] *
                           cpd_tags[prev_tag].prob(tag) * cpd_tagwords[tag].prob(sent[index]))
        current_viterbi[tag] = prev_viterbi[best_previous] * cpd_tags[best_previous].prob(tag) * \
                               cpd_tagwords[tag].prob(sent[index])
        current_backpointer[tag] = best_previous

    viterbi.append(current_viterbi)
    backpointer.append(current_backpointer)

    # 每次找完y把目前最好的打印一下
    current_best = max(current_viterbi.keys(), key=lambda tag: current_viterbi[tag])
    print('\nWord "' + sent[index] + '" current best two-tag sequence: ', current_backpointer[current_best], current_best)

# 找所有以END结束的tag sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(), key=lambda prev_tag: prev_viterbi[prev_tag] *
                    cpd_tags[prev_tag].prob('END'))
prob_tagsequence = prev_viterbi[best_previous] * cpd_tags[best_previous].prob('END')

# 这时我们是倒着存的，因为好的在后面
best_tagsequence = ['END', best_previous]
# 同理这也倒过来
backpointer.reverse()

# 回溯
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]
# 显示结果
best_tagsequence.reverse()
print('The sentence was: ', end=' ')
for word in sent:
    print(word, end=' ')
print('\n')
print('The best tag sequence is: ', end=' ')
for tag in best_tagsequence:
    print(tag, end=' ')
print('\n')
print('The probability of the best tag sequence is: ', prob_tagsequence)
# 结果不对，主要是语料不够大
