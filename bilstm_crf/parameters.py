# -*- coding: utf-8 -*-

UNIT = 'word'  # unit of tokenization (char, word)
RNN_TYPE = 'LSTM'  # LSTM or GRU
NUM_DIRS = 2  # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 256
EMBED = ['char', 'word']  # embeddings (char, word)
EMBED_SIZE = 300
HIDDEN_SIZE = 1000
DROPOUT = 0.5
LEARNING_RATE = 1e-4
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

KEEP_IDX = False  # use the existing indices when preparing addtional data
NUM_DIGITS = 4  # number of digits to print
