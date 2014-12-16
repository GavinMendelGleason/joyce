#!/usr/bin/env python

"""
Run various different authorship analyses and create plots.  

"""

import subprocess

default_params = ["./joyce.py","--train", 
                  "--examine", "/home/rowan/lib/Joycechekovgavin/disputed/politics-and-cattle-disease-1912.txt",
                  "--log", "./joyce.log"]

word_ngrams = [(x,y) for x in range(1,4) for y in range(x,4)]
char_ngrams = [(x,y) for x in range(3,5) for y in range(x,5)]

algorithms = [['--alg','LinearSVC', "--params", "{}"],
              ['--alg', 'SVC', "--params", "{'probability' : True}"],
              ['--alg','NuSVC', "--params", "{'probability' : True}"]]

for alg_params in algorithms:
    for pair in word_ngrams:
        # Basic ngram style analysis
        subprocess.call(default_params + alg_params + 
                        ["--ngrams", str(pair)])
    for pair in char_ngrams: 
        subprocess.call(default_params + alg_params + 
                        ["--ngrams", str(pair), 
                         "--analyser", "char_wb"])
