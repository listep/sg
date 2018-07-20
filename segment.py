import sys
from pyhanlp import HanLP

with open(sys.argv[1], "r") as f:
    idx = 0
    for line in f:
        words = []
        for term in HanLP.segment(line.strip()):
            words.append(term.word)
        print(" ".join(words))
