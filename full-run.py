#!/usr/bin/env python

"""
Run various different authorship analyses and create plots.  

"""

import subprocess
import logging 
import argparse 

__LOG_PATH__ = './full-run.log'
__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
if __name__ == '__main__': 
    """We will need some code for populating tables from command line 
       here, possibly by sending in some basic parameters and a glob.
    """    
    parser = argparse.ArgumentParser(description='Run loop over multiple parameters in search for best identification approach.')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    args = vars(parser.parse_args())

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(filename=args['log'],level=logging.INFO,
                        format=__LOG_FORMAT__)

    # no longer using examine, tests are stored in DB
    #"--examine", "/home/rowan/lib/Joycechekovgavin/disputed/politics-and-cattle-disease-1912.txt",

    default_params = ["./joyce.py","--train",                       
                      "--log", args['log']]
    
    word_ngrams = [(x,y) for x in range(1,4) for y in range(x,4)]
    char_ngrams = [(x,y) for x in range(3,5) for y in range(x,5)]

    algorithms = [['--alg','LinearSVC', "--params", "{}"],
                  ['--alg', 'SVC', "--params", "{'probability' : True}"],
                  ['--alg','NuSVC', "--params", "{'probability' : True}"]]

    seg_sizes = [x for x in xrange(500,2000,500)]

    logging.info("#" * 80)

    for seg_size in seg_sizes:

        logging.info("Segment size: %s" % seg_size)

        subprocess.call(['./metadata.py', 
                         # change segmentation size
                         '--seg-size', str(seg_size)]) 

        for alg_params in algorithms:
            for pair in word_ngrams:
                # Basic ngram style analysis
                subprocess.call(default_params + alg_params + 
                                ["--ngrams", str(pair)])
            for pair in char_ngrams: 
                subprocess.call(default_params + alg_params + 
                                ["--ngrams", str(pair), 
                                 "--analyser", "char_wb"])
            
