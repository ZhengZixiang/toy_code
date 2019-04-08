# -*- coding: utf-8 -*-
import sys

from utils import *
from parameters import *


def load_data():
    data = []
    if KEEP_IDX:
        char2id = load_tkn_to_idx(sys.argv[1] + '.char2id')
        word2id = load_tkn_to_idx(sys.argv[1] + '.word2id')
        tag2id = load_tkn_to_idx(sys.argv[1] + '.tag2id')
    else:
        char2id = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        word2id = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tag2id = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
        # IOB tags
        tag2id['B'] = len(tag2id)
        tag2id['I'] = len(tag2id)
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            tokens = line.split(' ')
            seq = []
            tags = []
            # for tkn in tokens:
            for word in tokens:
                # word, tag = re.split('/(?=[^/]+$)', tkn)
                # word = normalize(word)
                if not KEEP_IDX:
                    if word not in word2id:
                        word2id[word] = len(word2id)
                    # if tag not in tag2id:
                    #     tag2id[tag] = len(tag2id)
                    for c in word:
                        if c not in char2id:
                            char2id[c] = len(char2id)
                ctags = ['B' if i == 0 else 'I' for i in range(len(word))]
                seq.extend([str(char2id[c]) if c in char2id else str(UNK_IDX) for c in word])
                tags.extend([str(tag2id[t]) for t in ctags])
            data.append(seq + tags)
        data.sort(key=lambda x: -len(x))
    return data, char2id, word2id, tag2id


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: %s training_data' % sys.argv[0])
    data, char2id, word2id, tag2id = load_data()
    save_data(sys.argv[1] + '.csv', data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + '.char2id', char2id)
        save_tkn_to_idx(sys.argv[1] + '.word2id', word2id)
        save_tkn_to_idx(sys.argv[1] + '.tag2id', tag2id)
