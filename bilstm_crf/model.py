# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from embedding import Embedding
from parameters import *


CUDA = torch.cuda.is_available()
torch.manual_seed(12345)


class RNN_CRF(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, num_tags):
        super(RNN_CRF, self).__init__()
        self.rnn = RNN(char_vocab_size, word_vocab_size, num_tags)
        self.crf = CRF(num_tags)
        self = self.cuda() if CUDA else self

    def forward(self, cx, wx, y):
        mask = wx.data.gt(0).float()
        h = self.rnn(cx, wx, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    def decode(self, cx, wx):
        """for prediction"""
        mask = wx.data.gt(0).float()
        h = self.rnn(cx, wx, mask)
        return self.crf.decode(h, mask)


class RNN(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, num_tags):
        super(RNN, self).__init__()

        # architecture
        self.embed = Embedding(char_vocab_size, word_vocab_size, EMBED_SIZE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = NUM_DIRS == 2
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags)  # RNN output to tag

    def init_hidden(self):
        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)  # hidden state
        if RNN_TYPE == 'LSTM':
            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)  # cell state
            return (h, c)
        return  h

    def forward(self, cx, wx, mask):
        self.hidden = self.init_hidden()
        x = self.embed(cx, wx)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h


class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000.  # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000.  # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000.  # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000.  # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, y, mask):
        """forward algorithm"""
        # initialize forward variable in log space
        score = Tensor(BATCH_SIZE, self.num_tags).fill_(-10000.)  # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0)  # [1, C, C]
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score  # partition function

    def score(self, h, y, mask):
        """calculate the score of a given sequence"""
        score = Tensor(BATCH_SIZE).fill_(0.)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t+1]]  for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t+1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask):
        """Viterbi decoding"""
        bptr = LongTensor()
        score = Tensor(BATCH_SIZE, self.num_tags).fill_(-10000.)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(BATCH_SIZE):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
