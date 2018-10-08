# -*- coding: utf-8 -*-
import os
import sys
import logging
import jieba

# command example:
# python segment_text.py wiki.zh.text wiki.zh.text.seg
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    i = 0

    output = open(outp, 'w')
    jieba.enable_parallel()
    with open(inp, 'r') as f:
        for line in f.readlines():
            i = i + 1
            words = jieba.cut(line)
            output.write(' '.join(words))
            if i % 10000 == 0:
                logger.info('Segmented ' + str(i) + ' articles')
        output.close()
        logger.info('Finished Segmented ' + str(i) + ' articles')
    f.close()