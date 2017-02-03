#!/usr/bin/env python3
import argparse
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="word2vec model path")
    parser.add_argument("format", help="1 = binary format, 0 = text format", type=int)
    parser.add_argument("k", help="number of clusters", type=int)
    parser.add_argument("output", help="output file")
    args = parser.parse_args()

    return

if __name__ == "__main__":
    main()


