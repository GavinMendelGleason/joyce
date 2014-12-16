#!/usr/bin/env python

"""
Run various different authorship analyses and create plots.  

"""

import subprocess



# Basic ngram style analysis
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(1,3)",
                 "--params", "{'probability' : True}", 
                 "--alg", "NuSVC", 
                 "--log", "./joyce.log"])
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(1,3)",
                 "--params", "{'probability' : True}", 
                 "--alg", "SVC", 
                 "--log", "./joyce.log"])
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(1,3)",
                 "--params", "{}", 
                 "--alg", "LinearSVC", 
                 "--log", "./joyce.log"])
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(4,4)",
                 "--params", "{'probability' : True}", 
                 "--analyser", "char_wb",
                 "--alg", "NuSVC", 
                 "--log", "./joyce.log"])
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(4,4)",
                 "--params", "{'probability' : True}", 
                 "--analyser", "char_wb",
                 "--alg", "SVC", 
                 "--log", "./joyce.log"])
subprocess.call(["./joyce.py","--train", 
                 "--ngrams", "(4,4)",
                 "--params", "{}", 
                 "--analyser", "char_wb",
                 "--alg", "LinearSVC", 
                 "--log", "./joyce.log"])

# Use best for examination.
subprocess.call(["./joyce.py","--train", 
                 "--examine", "/home/rowan/lib/Joycechekovgavin/disputed/politics-and-cattle-disease-1912.txt",
                 "--ngrams", "(4,4)",
                 "--params", "{}", 
                 "--analyser", "char_wb",
                 "--alg", "LinearSVC", 
                 "--log", "./joyce.log"])

