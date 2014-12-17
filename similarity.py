#!/usr/bin/env python

"""
Plot average similiarty of each document of each positive segment with all negative segments versus, and a plot of all positive segments versus all positive segments.

We will use a few similiarty measures including average 
"""

import subprocess
import logging 
import scipy.spatial.distance as distance
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
import ast 
import metadata 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import cPickle as pickle

def reorderRows(indexes, matrix): 
    """Creates a new matrix, reordering rows such that the 'front' rows appear first."""
    m = sparse.dok_matrix(matrix)
    (row,column) = m.nonzero()
    shape = matrix.shape
    new = sparse.dok_matrix(shape)
    newi = 0
    (row,column) = m.nonzero()
    for i in indexes:         
        for j in column: 
            new[(newi,j)] = m[(i,j)]
        newi+=1
    return sparse.csr_matrix(new)

def run3Dheatmap(X):
    
    plt.imshow(data, interpolation='none', aspect=3./20)

    plt.xticks(range(3), ['a', 'b', 'c'])

    plt.jet()
    plt.colorbar()

    plt.show() 

__LOG_PATH__ = './full-run.log'
__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
if __name__ == '__main__': 
    """We will need some code for populating tables from command line 
       here, possibly by sending in some basic parameters and a glob.
    """    
    parser = argparse.ArgumentParser(description='Check inter document similiarity.')
    parser.add_argument('--ngrams', help="Pair of (min, max) n-gram size", default='(1,2)')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--analyser', help="Type of tokenisation, one of [word,char_wb]", default='word')
    parser.add_argument('--create', help='Create the reordered array', action='store_const', const=True, default=False)
    parser.add_argument('--backing-store', help='The reordered matrix backing store', default='./reordered.pkl')
    args = vars(parser.parse_args())

    args['ngrams'] = ast.literal_eval(args['ngrams'])

    if args['create']:
        segments = metadata.get_all_segments()
        # joyce
        features = metadata.get_all_author_or_not_features('James Joyce')

        vec = CountVectorizer(min_df=1, ngram_range=args['ngrams'], analyzer=args['analyser'])
        counts = vec.fit_transform(segments)
    
        transformer = TfidfTransformer() 
        tfidf = transformer.fit_transform(counts)       

        negative_idx = []
        positive_idx = []
        for i in range(0,len(segments)): 
            if features[i] == 0: 
                negative_idx.append(i)
            else: 
                positive_idx.append(i)
    
        tfidf_reordered = reorderRows(negative_idx+positive_idx, tfidf)
        f = open(args['backing_store'],'wb')
        pickle.dump(tfidf_reordered,f)

    f = open(args['backing_store'],'rb')
    pickle.load(f)
    # calculate lower half triangle
    Y = distance.pdist(tfidf_reordered)
    run3Dhisto(Y)                

        

