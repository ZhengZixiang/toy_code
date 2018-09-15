# -*- coding: utf-8 -*-
import sys


if __name__ == '__main__':
    handler = sys.stdin
    for line in handler:
        if not line:
            continue
        terms = line.strip().split(' ')
        for i in terms:
            print(i)
