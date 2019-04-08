# -*- coding: utf-8 -*-
import re
import torch


def normalize(x):
    x = re.sub('\s+', ' ', x)
    x = re.sub('^ | $', '', x)
    x = x.lower()
    return x


def tokenize(x, unit):
    x = normalize(x)
    if unit == 'char':
        return re.sub(' ', '', x)
    if unit == 'word':
        return x.split(' ')


def save_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for seq in data:
            f.write(' '.join(seq) + '\n')


def load_tkn_to_idx(filename):
    print('loading %s' % filename)
    tkn2id = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]  # -1: \n
            tkn2id[line] = len(tkn2id)
    return tkn2id


def load_idx_to_tkn(filename):
    print('loading %s' % filename)
    id2tkn = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]  # -1: \n
            id2tkn.append(line)
    return id2tkn


def save_tkn_to_idx(filename, tkn2id):
    with open(filename, 'w', encoding='utf-8') as f:
        for tkn, _ in sorted(tkn2id.items(), key=lambda x: x[1]):
            f.write('%s\n' % tkn)


def load_checkpoint(filename, model=None):
    print('loading %s' % filename)
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('saved model: epoch = %d, loss = %f' % (checkpoint['epoch'], checkpoint['loss']))
    return epoch


def save_cehckpoint(filename, model, epoch, loss, time):
    print('epoch = %d, loss = %f, time = %f' % (epoch, loss, time))
    if filename and model:
        print('saving %s' % filename)
        checkpoint = dict()
        checkpoint['statedict'] = model.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['loss'] = loss
        torch.save(checkpoint, filename + '.epoch%d' % epoch)
        print('saved model at epoch %d' % epoch)


def iob2txt(x, y, unit):
    out = ''
    x = tokenize(x, unit)
    for i, j in enumerate(y):
        if i and j[0] == 'B':
            out += ' '
        out += x[i]
    return out


def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0
