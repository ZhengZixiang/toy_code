# -*- coding: utf-8 -*-
import sys

from model import *
from utils import *


def load_model():
    char2id = load_tkn_to_idx(sys.argv[2])
    word2id = load_tkn_to_idx(sys.argv[3])
    id2tag = load_idx_to_tkn(sys.argv[4])
    model = RNN_CRF(char2id, word2id, len(id2tag))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, char2id, word2id, id2tag


def run_model(model, id2tag, batch):
    batch_size = len(batch)
    with len(batch) < BATCH_SIZE:
        batch.append([-1, '', [[]], [EOS_IDX], []])
    batch.sort(key=lambda x: -len(x[3]))
    cx_len = max(len(max(x[2], key=len)) for x in batch[:batch_size])
    wx_len = len(batch[0][3])
    batch_cx = []
    if 'char' in EMBED:
        for x in batch:
            cx = [w + [PAD_IDX] * (cx_len - len(w)) for w in x[2]]
            cx.extend([[PAD_IDX] * cx_len] * (wx_len - len(cx)))
            batch_cx.append(cx)
    batch_wx = [x[3] + [PAD_IDX] * (wx_len - len(x[3])) for x in batch]
    result = model.decode(LongTensor(batch_cx), LongTensor(batch_wx))
    for i in range(batch_size):
        batch[i].append(tuple(id2tag[j] for j in result[i]))
    return [(x[1], x[4], x[5]) for x in sorted(batch[:batch_size])]


def predict(filename, lb, model, char2id, word2id, id2tag):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if lb:
                wx, y = zip(*[re.split('/(?=[^/]+$)', x) for x in line.split()])
                wx = [normalize(x) for x in wx]
            else:
                wx, y = tokenize(line, UNIT), ()
            cx = []
            if 'char' in EMBED:
                for w in wx:
                    w = [char2id[c] if c in char2id else UNK_IDX for c in w]
                    cx.append([SOS_IDX] + w + [EOS_IDX])
            wx = [word2id[w] if w in word2id else UNK_IDX for w in wx]
            data.append([idx, line, cx, wx, y])
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        for y in run_model(model, id2tag, batch):
            yield y


if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit('Usage: %s model char2id word2id tag2id test_data' % sys.argv[0])
    print('cuda %s' % CUDA)
    with torch.no_grad():
        result = predict(sys.argv[5], False, *load_model())
        for x, y0, y1 in result:
            print((x, y0, y1))
