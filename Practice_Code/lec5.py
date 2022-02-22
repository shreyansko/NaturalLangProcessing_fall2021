#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:38:28 2021

@author: shreyanskothari
"""

#import nltk
from utils.my_funcs import *
import pickle


the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/corpuses"
#my_pd = file_seeker(the_path, False)

#my_pd['body_sw'] = my_pd.body.apply(rem_sw)

#Saving my_pd as a pickle object
#pickle.dump(my_pd, open("my_pd.pkl", "wb")) #wb = write binary
#Opening it again and saving in test
my_pd = pickle.load(open("my_pd.pkl", "rb")) #rb = read binary
    

my_pd['sw_stem'] = my_pd.body_sw.apply(stem_fun)
pickle.dump(my_pd, open("my_pd.pkl", "wb"))
#dictionary = nltk.corpus.words.words('en')
#my_pd['sw_dictionary'] = my_pd.body_sw.apply(check_dictionary)