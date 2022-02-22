#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:18:30 2021

@author: shreyanskothari
"""
from crawler import write_crawl_results

the_data = write_crawl_results(["NFL football", "NHL hockey"], 20)




from utils.my_functions import *
import nltk
nltk.download('stopwords')
the_data['body_sw'] = the_data.body_basic.apply(rem_sw)

the_data['body_stem'] = the_data.body_sw.apply(stem_fun)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
import pandas as pd
my_pd =  pd.DataFrame(vectorizer.fit_transform(the_data.body_stem).toarray())
my_pd.index = the_data.label
my_pd.columns = vectorizer.vocabulary_

path_o = "/Users/shreyanskothari/Desktop/Natural Language Processing/Code"
my_pca = my_pca_fun(my_pd, 0.99, path_o)