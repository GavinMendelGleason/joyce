#!/usr/bin/env python

"""
Plot average similiarty of each document of each positive segment with all negative segments versus, and a plot of all positive segments versus all positive segments.

We will use a few similiarty measures including average 
"""

import subprocess
import logging 
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description='Allow command line uploading of files into the database.')
    parser.add_argument('--ngrams', help="Pair of (min, max) n-gram size", default='(1,2)')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)

    args['ngrams'] = ast.literal_eval(args['ngrams'])

    segments = metadata.get_all_segments()
    # joyce
    features = metadata.get_all_author_or_not_features('James Joyce')

    vec = CountVectorizer(min_df=1, ngram_range=(1,3), analyzer=args['analyser'])
    counts = vec.fit_transform(segments)
    
    transformer = TfidfTransformer() 
    tfidf = transformer.fit_transform(counts)       

    negative_tfidf = []
    positive_tfidf = []

    for i in range(0,len(segments)): 
        if features[i] == 0: 
            negative_tfidf.append(tfidf[i])
        else: 
            positive_tfidf.append(tfidf[i])
    
    # calculate lower half triangle
    Y = distance.pdist(negative_tfidf)
    run3Dhisto(Y)                

        

