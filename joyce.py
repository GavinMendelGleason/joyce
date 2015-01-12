#!/usr/bin/env python

import argparse 
import logging 
import glob
from os import getenv, path
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
import metadata
import cPickle as pickle
#import pickle
import numpy as np
import scipy 
import ast

def document(path): 
    with open (path, "r") as myfile:
        data=myfile.read().replace('\n', '')
    return data 

def create_corpus(files): 
    documents = []
    for p in files: 
        documents.append(document(p))
    # need to keep track of the source.
    return documents

# Using scikit learn instead    
def xvalidation_subsets(tfidf,features,k): 
    block = []
    length = tfidf.shape[0]
    size =  length / k 
    subsets = []
    ## lost tooth approach
    for i in xrange(size, length-size, size): 
        new_tfidf = scipy.sparse.coo_matrix.tocsc(np.concatenate([(tfidf[0:i]).toarray(),(tfidf[i+size:length]).toarray()],axis=0))
        new_features = features[0:i] + features[i+size:length]
        subsets.append((new_tfidf,new_features))

    return subsets

__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
__LIB_PATH__ = getenv("HOME") + '/lib/Joycechekovgavin/'
__DEFAULT_PATH__ = __LIB_PATH__ + 'corpus/'
__LOG_PATH__ = __LIB_PATH__ + 'joyce.log'
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Playing with word embeddings.')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--train', help='Start training from DB', action='store_const', const=True, default=False)
    # Should we have this available for a single examination of a document?
    parser.add_argument('--examine', help='Examine a document', default=None)
    parser.add_argument('--backing-store', help='Path of svm backing store', default='backing_store.pkl')
    parser.add_argument('--cross-validations', help="number of cross-validation segments", default="5")
    parser.add_argument('--alg', help="One of [SVC, LinearSVC, NuSVC]", default ='NuSVC')
    parser.add_argument('--params', help="Algorithm parameters as dictionary string", default="{}")
    parser.add_argument('--ngrams', help="Pair of (min, max) n-gram size", default='(1,2)')
    parser.add_argument('--analyser', help="Type of tokenisation, one of [word,char_wb]", default='word')
    args = vars(parser.parse_args())
    
    # evaluate commandline params of python objects
    args['params'] = ast.literal_eval(args['params'])
    args['ngrams'] = ast.literal_eval(args['ngrams'])

    # set up logging
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(filename=args['log'],level=logging.INFO,
                        format=__LOG_FORMAT__)
    
    logging.info(str(args))

    #files = joyce + other
    if args['train']: 
        segments = metadata.get_training_segments()
        logging.info("# of Segments: %s" % len(segments))

        features = metadata.get_training_author_or_not_features('James Joyce')
        #print "# of Features: %s" % len(features) 

        vec = CountVectorizer(min_df=1, ngram_range=args['ngrams'], analyzer=args['analyser'])
        counts = vec.fit_transform(segments)
        
        transformer = TfidfTransformer() 
        tfidf = transformer.fit_transform(counts) # counts
        
        if args['alg'] == 'SVC': 
            svm = SVC(**args['params'])
        elif args['alg'] == 'NuSVC':
            svm = NuSVC(**args['params'])
        elif args['alg'] == 'LinearSVC': 
            svm = LinearSVC(**args['params'])

        features = np.array(features)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(tfidf,features,test_size=0.4, random_state=0)
        clf = svm.fit(X_train, y_train)
        score = clf.score(X_test, y_test)  
        cross = cross_validation.cross_val_score(svm, tfidf, y=features, cv=5)

        logging.info("Score: %s" % score)
        logging.info("Five way cross validation: %s" % cross)
        logging.info("Mean of cross validation: %s" % np.mean(cross))
        logging.info("Variance of cross validation: %s" % np.var(cross))
        f = open(args['backing_store'], 'wb')
        p = pickle.Pickler(f) 
        p.fast = True
        p.dump((vec,transformer,svm)) # d is your dictionary
        f.close()

        if args['degradation']: 
            """We are going to attempt degredation of results"""
            pass

    if path.isfile(args['backing_store']):        
        f = open(args['backing_store'], 'rb')
        (vec,transformer,svm) = pickle.load(f)
        f.close()
    else: 
        sys.exit("Unable to load backing store")

    # This is just for the craic at this stage.
    # we need to be putting this in the DB...
    # if args['examine']: 
    #     # not yet implemented
    #     s = metadata.read_file(args['examine'])
    #     segments = metadata.chunkify(s,args['seg_size'])        
    # else: 

    segments = metadata.get_test_segments()
    counts = vec.transform(segments) 
    tfidf = transformer.transform(counts).toarray()
    results = svm.predict(tfidf)
    proba = None
    if 'predict_proba' in dir(svm): 
        proba = svm.predict_proba(tfidf)
    pos = filter(lambda x: x==1, results)
    neg = filter(lambda x: x==0, results)
    logging.info("# pos: %s" % len(pos))
    logging.info("# neg: %s" % len(neg))
    logging.info("Segments classified: %s %%" % (len(pos) / float(len(results)) * 100))
    try: 
        if proba <> None: 
            no = [x for [x, y] in proba]
            yes = [y for [x, y] in proba]
            logging.info("Mean probability of classifications:  No: %s Yes: %s" % (np.mean(no), np.mean(yes)))
    except Exception as e: 
        logging.error(str(e))



    
