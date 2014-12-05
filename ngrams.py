#!/usr/bin/env python

class NGrams(object): 
    """NGrams will return all character n-grams of size N
       from a document."""
    def __init__(self, document, size=7, case_fold=False): 
        if size < 3: 
            raise Exception("You can't have n-grams smaller than 3")
        self.case_fold = case_fold
        self.document = document

    def __iter__(self): 
        return self

    # python 3
    def next(self): 
        return self.__next__()
        
    def process_token(self,token): 
        # For now, we only do case folding
        # but this monadic style means we can
        # slice tokens if needed, or add prefixes
        if self.casefold: 
            return [token.lower()]
        else:
            return [token]

    def __next__(self): 
        for line in self.document: 
            words = line.split(' ')
            ngrams = []
            for word in words: 
                for size=
    

class CorpusNGrams(object): 
    """Iterator that yields sentences from the corpus, one at a time so we don't need to load everything into memory"""

    def __init__(self,files): 
        self.file_queue = files
        self.last_line = None
        if self.file_queue != []:
            f = open(self.file_queue.pop(), 'rb')
            self.current_stream = Sentences(iter(lambda : f.read(1), ''))
        else: 
            logging.info("Please provide a list of files for parsing")
        
    def __iter__(self): 
        return self

    # Python 3
    def __next__(self): 
        return self.next()
        
    def next(self): 
        try: 
            return self.current_stream.next()
        except StopIteration: 
            if self.file_queue != []: 
                f = open(self.file_queue.pop(), 'rb')
                self.current_stream = NGrams(iter(lambda : f.read(1), ''))
                return self.next()
            else:
                raise StopIteration()
