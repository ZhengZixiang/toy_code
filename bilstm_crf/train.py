# -*- coding: utf-8 -*-
import os
import sys
import time

from model import *
from parameters import *
from utils import *


def load_data():
    data = []
    batch_cx = []  # character input
    batch_wx = []  # word input
    batch_y = []
    cx_maxlen = 0  # maximum length of character sequence
    wx_maxlen = 0  # maximum length of word sequence
    char2id = load_tkn_to_idx(sys.argv[2])
    id2word = load_idx_to_tkn(sys.argv[3])
    tag2id = load_tkn_to_idx(sys.argv[4])
    print('loading %s' % sys.argv[5])
    with open(sys.argv[5], 'r') as f:
        for line in f:
            line = line.strip()
            seq = [int(i) for i in line.split(' ')]
            wx_len = len(seq) // 2
            wx = seq[:wx_len]
            if not wx_maxlen:  # the first line is longest in its mini-batch
                wx_maxlen = wx_len
            wx_pad = [PAD_IDX] * (wx_maxlen - wx_len)
            batch_wx.append(wx + wx_pad)
            batch_y.append([SOS_IDX] + seq[wx_len:] + wx_pad)
            if 'char' in EMBED:
                cx = [[char2id[c] for c in id2word[i]] for i in wx]
                cx_maxlen = max(cx_maxlen, len(max(cx, key=len)))
                batch_cx.append([[SOS_IDX] + w + [EOS_IDX] for w in cx])
            if len(batch_wx) == BATCH_SIZE:
                if 'char' in EMBED:
                    for cx in batch_cx:
                        for w in cx:
                            w += [PAD_IDX] * (cx_maxlen - len(w) + 2)
                        cx += [[PAD_IDX] * (cx_maxlen + 2)] * ((wx_maxlen - len(cx)))
                data.append((LongTensor(batch_cx), LongTensor(batch_wx), LongTensor(batch_y)))
                batch_cx = []
                batch_wx = []
                batch_y = []
                cx_maxlen = 0
                wx_maxlen = 0
        print('data size: %d' % (len(data) * BATCH_SIZE))
        print('batch size: %d' % BATCH_SIZE)
        return data, len(char2id), len(id2word), len(tag2id)


def train():
    num_epochs = int(sys.argv[6])
    data, char_vocab_size, word_vocab_size, num_tags = load_data()
    model = RNN_CRF(char_vocab_size, word_vocab_size, num_tags)
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if os.path.isfile(sys.argv[1]) else 0
    filename = re.sub('\.epoch[0-9]+$', '', sys.argv[1])
    print('training model...')
    for e in range(epoch+1, epoch+num_epochs+1):
        loss_sum = 0
        timer = time.time()
        for cx, wx, y in data:
            model.zero_grad()
            loss = torch.mean(model(cx, wx, y))
            loss.backward()
            optim.step()
            loss = loss.item()
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if e % SAVE_EVERY and e != epoch + num_epochs:
            save_cehckpoint('', None, e, loss_sum, timer)
        else:
            save_cehckpoint(filename, model, e, loss_sum, timer)


if __name__ == '__main__':
    if len(sys.argv) != 7:
        sys.exit('Usage: %s model char2id word2id tag2id training_data.csv num_epoch' % sys.argv[0])
    print('cuda %s' % CUDA)
    train()

