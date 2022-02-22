#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:14:47 2021

@author: shreyanskothari
"""
import os
os.chdir('/Users/shreyanskothari/Desktop/Natural Language Processing/Code')
from utils.my_functions import *

import pickle
my_pd = pickle.load(open("/Users/shreyanskothari/Desktop/Natural Language Processing/code/utils/my_pd.pkl", "rb"))



def stem_fun(filein):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    my_text = [stemmer.stem(word) for word in filein.split()]
    my_text = ' '.join(my_text)
    return my_text


my_pd["stem_sw"] = 
my_pd["jaccard_similarity"] = my_pd.body.apply(
    lambda x: jaccard_fun(x, "machine learning is fun"))


