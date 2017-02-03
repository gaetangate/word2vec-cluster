#!/usr/bin/env python3
"""

WORD2VEC CENTROID TOOL

Author: Gaetano Rossiello, University of Bari, Italy
Email: gaetano.rossiello@uniba.it

"""

import argparse
import logging
import time
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="word2vec model path")
    parser.add_argument("format", help="1 = binary format, 0 = text format", type=int)
    parser.add_argument("k", help="number of clusters", type=int)
    parser.add_argument("output", help="output file")
    args = parser.parse_args()

    start = time.time()
    print("Load word2vec model ... ", end="", flush=True)
    w2v_model = Word2Vec.load_word2vec_format(args.model, binary=bool(args.format))
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)
    word_vectors = w2v_model.wv.syn0
    n_words = word_vectors.shape[0]
    vec_size = word_vectors.shape[1]
    print("#words = {0}, vector size = {1}".format(n_words, vec_size))

    start = time.time()
    print("Compute clustering ... ", end="", flush=True)
    kmeans = KMeans(n_clusters=args.k, n_jobs=-1, random_state=0)
    idx = kmeans.fit_predict(word_vectors)
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)

    start = time.time()
    print("Generate output file ... ", end="", flush=True)
    word_centroid_list = list(zip(w2v_model.wv.index2word, idx))
    word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
    file_out = open(args.output, "w")
    # file_out.write("WORD,K\n")
    for word_centroid in word_centroid_list_sort:
        line = word_centroid[0] + ',' + str(word_centroid[1]) + '\n'
        file_out.write(line)
    file_out.close()
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)

    return

if __name__ == "__main__":
    main()


