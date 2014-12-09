#!/usr/bin/env python

class NGrams(object): 
    """NGrams will return all character n-grams of size N
       from a document."""
    def __init__(self, document, size=7, case_fold=False): 
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
        if self.empty: 
            raise StopIteration()
        words = [self.left_padding]
        while True:
            # if this raises StopIteration() are we grand?
            try: 
                char = self.document.next()
                if char == r'.' or char == '?' or char == '!': 
                    if not self.stack == '':
                        words += self.process_token(self.stack)                
                        self.stack = ''
                    words.append(char)
                    return words + [self.right_padding]
                elif char == r' ' or char == '\n' or char == '\r': 
                    if not self.stack == '':
                        words += self.process_token(self.stack)
                        self.stack = ''
                else: 
                    self.stack += char
            except StopIteration as si: 
                if self.stack != '': 
                    words += self.process_token(self.stack)
                self.empty = True
                
                # if there is something to return, we stop iteration 
                # on the next go, otherwise stop now.
                if words != []:
                    return words + self.right_padding
                else:
                    raise StopIteration()
    
class SkipGrams(object): 
    """Take a document and create SkipGram pairs from it."""
    def __init__(self,sentence,window_size=2,left_padding='<s>',right_padding='</s>'):
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.left = []
        self.right = []
        self.sentence = sentence
        self.window_size = window_size
        self.current = -window_size
        self.length = len(self.sentence)

    def __iter__(self):
        return self

    # python 3
    def next(self): 
        return self.__next__()
    
    def __next__(self): 
        if self.current + self.window_size >= self.length:
            raise StopIteration()
        else:
            left_slice = [self.left_padding] * max(0, -self.current)
            right_slice = [self.right_padding] * max(0, self.current + self.window_size * 2 + 1 - self.length)
            left_start = max(0,self.current)            
            left_end = max(0,self.current+self.window_size)
            right_start = min(self.length-1, self.current+self.window_size+1)
            right_end = min(self.length-1, self.current+self.window_size*2+1)
            left_result = left_slice + self.sentence[left_start:left_end] 
            right_result = self.sentence[right_start:right_end] + right_slice
            print self.current
            self.current += 1
            return (left_result,right_result)

class Sentences(object): 
    """Sentences will return an iterator object 
    returning each sentence from the document 
    (which is a character stream) 
    as a token list in turn.
    """
    def __init__(self, document, casefold=True): 

        if type(document) == str: 
            document = iter(document)

        self.words = []
        self.stack = ''
        self.document = document
        self.empty = False
        self.casefold = True

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
        if self.empty: 
            raise StopIteration()
        words = []
        while True:
            # if this raises StopIteration() are we grand?
            try: 
                char = self.document.next()
                if char == r'.' or char == '?' or char == '!': 
                    if not self.stack == '':
                        words += self.process_token(self.stack)                
                        self.stack = ''
                    words.append(char)
                    return words
                elif char == r' ' or char == '\n' or char == '\r': 
                    if not self.stack == '':
                        words += self.process_token(self.stack)
                        self.stack = ''
                else: 
                    self.stack += char
            except StopIteration as si: 
                if self.stack != '': 
                    words += self.process_token(self.stack)
                self.empty = True
                
                # if there is something to return, we stop iteration 
                # on the next go, otherwise stop now.
                if words != []:
                    return words
                else:
                    raise StopIteration()

class CorpusSentences(object): 
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
                self.current_stream = Sentences(iter(lambda : f.read(1), ''))
                return self.next()
            else:
                raise StopIteration()
