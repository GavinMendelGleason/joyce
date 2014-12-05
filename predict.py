#!/usr/bin/env python

import theano.tensor as T
import time 
import cPickle 
import metadata
from gensim.models import Word2Vec
from tokenise import Sentences, SkipGram

"""
This programme will use a 3 layer system in an attempt to guess the words in a corpus based on context.  The idea being that if it is trained well on a particular author it can be discriminative on those texts which are not going to be representative.  Initial input vectors will come from word embeddings trained on a much large corpus.
"""


## First, structure 

class MLPNNLM(object): 
    def __init__(self, X, y, word_size = 100, 
                 window_size=3, learning_rate=0.001, 
                 hidden_size=100, backing_store=None, 
                 n_epochs=100, n_train_batches=100, 
                 batch_size=100
             ): 
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.n_train_batches=n_train_batches
        if self.backing_store:
            self.backing_store = backing_store
        else: 
            
        self.X_train, X_pretest, self.y_train, y_pretest = cross_validation.train_test_split(X,y,test_size=0.6, random_state=0)
        self.X_test, self.X_valid, self.y_test, self.y_valid = cross_validation.train_test_split(X_pretest,y_pretest,test_size=0.5, random_state=0)

        self.word_size = word_size
        self.layer1_size = window_size * self.word_size
        self.layer2_size = hidden_size
        self.layer3_size = self.word_size

        self.W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (layer1_size + layer2_size)),
            high=numpy.sqrt(6. / (layer1_size + layer2_size)),
            size=(layer1_size, layer2_size)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W')
        self.b1_values = = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / layer2_size), 
            high=numpy.sqrt(6. / layer2_size),
            size=(layer2_size,)), dtype=theano.config.floatX)

        self.b1 = theano.shared(value=b1_values, name='b1')

        self.U_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (layer2_size + layer3_size)),
            high=numpy.sqrt(6. / (layer2_size + layer3_size)),
            size=(layer2_size,label_dim)), dtype=theano.config.floatX)

        g = T.tanh(T.dot(x,W) + b1) 
        h = T.dot(g,U) + b2 ## No activation function for the output?

        negative_log_likelihood = T.sum((y - h)**2)
        y_pred = T.argmax(h,axis=1)

        # This is wrong - needs to be squared euclidean distance
        errors = T.mean(T.neq(y_pred, y))

        g_W = T.grad(cost=cost, wrt=W)
        g_b1 = T.grad(cost=cost, wrt=b1)
        g_U = T.grad(cost=cost, wrt=U) 
        g_b2 = T.grad(cost=cost, wrt=b2) 

        self.updates = updates = [(W, W - learning_rate * g_W),
                                  (b1, b1 - learning_rate * g_b1), 
                                  (U, U - learning_rate * g_U), 
                                  (b2, b2 - learning_rate * g_b2)]
        
        self.train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={
                                      x: train_x[index * self.batch_size:(index + 1) * self.batch_size],
                                      y: train_y[index * self.batch_size:(index + 1) * self.batch_size]})

        self.test_model = theano.function(inputs=[index],
                                 outputs=errors,
                                 givens={
                                     x: test_x[index * self.batch_size: (index + 1) * self.batch_size],
                                     y: test_y[index * self.batch_size: (index + 1) * self.batch_size]})

        self.validate_model = theano.function(inputs=[index],
                                     outputs=errors,
                                     givens={
                                         x: valid_x[index * self.batch_size: (index + 1) * self.batch_size],
                                         y: valid_y[index * self.batch_size: (index + 1) * self.batch_size]})

    def load(self):
        f = open(self.backing_store,'rb')
        tmp_dict = cPickle.load(f)
        f.close()          
        
        self.__dict__.update(tmp_dict) 

    def save(self):
        f = open(self.backing_store,'wb')
        cPickle.dump(self.__dict__,f,2)
        f.close()
        
    def train_nn(self): 
        """Train the parameters for the neural network. """

        patience = 100000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)

        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()
            
        done_looping = False
        epoch = 0
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
                #print "Minibatch: %s" % minibatch_index
                minibatch_avg_cost = train_model(minibatch_index)
                #print "Minibatch avg cost: %s" % minibatch_avg_cost
                # iteration number
                iter = (epoch - 1) * sefl.n_train_batches + minibatch_index
                #print "Validation: %s" % validation_frequency
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                       for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        ## Here we should dump best parameters.
                        self.save()

                        print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
              (best_validation_loss * 100., test_score * 100.))
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))


__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
__LIB_PATH__ = getenv("HOME") + '/lib/nn-ling/'
__DEFAULT_PATH__ = __LIB_PATH__ + 'corpus/'
__LOG_PATH__ = __LIB_PATH__ + 'nn-ling.log'
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Run NNLM on pre-trained word embeddings')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--learning-rate', help='Learning rate', default="0.001")
    parser.add_argument('--batch-size', help='Batch size', default="100")
    parser.add_argument('--hidden-layer-size',help='Size of the hidden layer', default = "50")   
    parser.add_argument('--window-size',help='Size of the before/after skipgram', default = "2")
    parser.add_argument('--n-train-batches', help='Number of training batches to process', default= "10")
    parser.add_argument('--n-epochs', help='Number of epochs', default = "100")
    parser.add_argument('--backing-store', help='Name of NNLM pickle file', default='./nnlm.pkl')
    parser.add_argument('--resurrect', help='Resurrect NNLM from pickle file', action="store_const", const=True)
    parser.add_argument('--word-embeddings', help='File containing word embeddings in word2vec format', default='./word-embeddings.w2v')
    parser.add_argument('--token-type', help='A type of token to use, one of (char,word)')
    args = vars(parser.parse_args())

    w2v_model = Word2Vec.load(args['word_embeddings'])
    word_size = w2v_model.size

    segments = metadata.get_training_segments()
    features = metadata.get_joyce_or_not_features()
    # First we have to prep the segments using the word_2_vec and parsing approach
    for segment in segments: 
        matrix = []
        for sentence in Sentences(parse_segment):
            for windows in SkipGram(sentence,args['window_size']): 
                vectors = []
                for gram in skip_gram: 
                    vectors.append(w2v_model[gram])
                matrix.append(numpy.concatenate(vectors))
                
    mlpnnlm = MLPNNLM(segments,features,
                      backing_store=args['backing_store'], 
                      batch_size=args['batch_size'],
                      n_train_batches=int(args['n_train_batches']), 
                      n_epochs=int(args['n_epochs']), 
                      
                  )
    
    mlpnnlm.train()
    
    mlpnnlm 
