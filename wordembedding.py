#!/usr/bin/env python
"""
Ideally we would use word vectors from a very large corpus of english words 
such as the one given by Google.

Currently simply reusing the corpus of segments.
"""
import argparse
from gensim.models import Word2Vec
from tokenise import Sentences, SkipGrams
import logging
import metadata
import numpy 
import dbm 
import UserDict
import cPickle as pickle

class DBMVectors(UserDict.DictMixin): 
    def __init__(self,backing_store=None, source_file=None): 
        self.source_file = source_file
        self.backing_store = backing_store
        if source_file and backing_store:
            self.init_dict()
        if backing_store: 
            self.resurrect_dict()

    def init_dict(self): 
        self.store = dbm.open(self.backing_store,'n')

        with open(self.source_file, 'rb') as f: 
            for l in f: 
                lst = l.split()
                word = lst[0]
                self.layer1_size = len(lst[1:])
                vec = numpy.zeros(self.layer1_size)
                i = 0
                for num in lst[1:]: 
                    vec[i] = float(num) 
                    i += 1
                self.store[word] = pickle.dumps(vec)
        self.store.close()
        self.store = dbm.open(self.backing_store, 'r')

    def resurrect_dict(self): 
        self.store = dbm.open(self.backing_store, 'r')
        # We have to hope that 'the' is in there!
        self.layer1_size = self['the'].shape[0]

    def __getitem__(self,key): 
        return pickle.loads(self.store[key])

    def __setitem__(self): 
        raise Exception("setting items not implemented.")

    def __delitem__(self): 
        raise Exception("deleting items not implemented.")

    def close(self): 
        self.store.close()

    def __del__(self):
        self.close()

    def keys(self):
        raise Exception("Keys not implemented.")


def createDocumentMatrices(segments,w2v_model): 
    features = []
    matrix = []
    zero = numpy.zeros(w2v_model.layer1_size)
    for segment in segments: 
        for sentence in Sentences(iter(segment)):
            vectors = []
            for gram in sentence:
                try: 
                    vectors.append(w2v_model[gram])
                except Exception as e: 
                    vectors.append(zero)
            matrix.append(numpy.concatenate(vectors))

    return matrix



def get_vec(w2v_model,gram): 
    if gram in w2v_model: 
        return w2v_model[gram]
    else:
        return numpy.zeros(w2v_model.layer1_size)

def createDocumentSkipMatrices(segments,w2v_model, window_size=3): 
    features = []
    matrix = []
    for segment in segments: 
        for sentence in Sentences(iter(segment)):
            for (left,word,right) in SkipGrams(sentence,window_size): 
                vectors = []
                for gram in left+right: 
                    vectors.append(get_vec(w2v_model,gram))
                matrix.append(numpy.concatenate(vectors))
                features.append(get_vec(w2v_model,word))

    return (matrix,features)

__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
__LOG_PATH__ = './joyce-predict.log'
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train word embeddings')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--ngram-type', help='Either word or char n-gram', default='word')
    parser.add_argument('--file', help='File name of w2v backing store', default='words.w2v')
    args = vars(parser.parse_args())

    # set up logging
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(filename=args['log'],level=logging.INFO,
                        format=__LOG_FORMAT__)

    segments = metadata.get_all_segments()
    
    model = None
    for segment in segments: 
        if model: 
            test = Sentences(iter(segment))
            model.train(Sentences(iter(segment)))
        else: 
            model = Word2Vec(Sentences(iter(segment)), size=150, window=3, min_count=2, workers=4)

    model.save(args['file'])
    
    
