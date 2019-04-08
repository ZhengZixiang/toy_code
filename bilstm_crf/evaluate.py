# -*- coding: utf-8 -*-
from collections import defaultdict
from predict import *


def evaluate(result):
    avg = defaultdict(float)
    tp = defaultdict(int)  # true positive
    tpfn = defaultdict(int)  # true positive + false negative
    tpfp = defaultdict(int)  # true positive + false positive
    for _, y0, y1 in result:  # actual value, prediction
        for y0, y1 in zip(y0, y1):
            tp[y0] += y0 == y1
            tpfn[y0] += 1
            tpfp[y1] += 1
    for y in sorted(tpfn.keys()):
        precision = tp[y] / tpfp[y] if tpfp[y] else 0
        recall = tp[y] / tpfn[y] if tpfn[y] else 0
        avg['macro_precision'] += precision
        avg['macro_recall'] += recall
        print()
        print('label = %s' % y)
        print('precision = %f (%d/%d)' % (precision, tp[y], tpfp[y]))
        print('recall = %f (%d/%d)' % (recall, tp[y], tpfn[y]))
        print('F1 = %f' % f1(precision, recall))
    avg['macro_precision'] /= len(tpfn)
    avg['macro_recall'] /= len(tpfn)
    avg['micro_precision'] = sum(tp.values() / sum(tpfp.values()))
    avg['micro_recall'] = sum(tp.values() / sum (tpfn.values()))
    print()
    print('macro precision = %f' % avg['macro_precision'])
    print('macro recall = %f' % avg['macro_recall'])
    print('macro f1 = %f' % f1(avg['macro_pr'], avg['macro_recall']))
    print()
    print('micro precision = %f' % avg['micro_precision'])
    print('micro recall = %f' % avg['micro_recall'])
    print('micro f1 = %f' % f1(avg['micro_pr'], avg['micro_recall']))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit('Usage: %s model char2id word2id tag2id test_data' % sys.argv[0])
    print('cuda %s' % CUDA)

    with torch.no_grad():
        evaluate(predict(sys.argvv[5], True, *load_model()))