#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:59:51 2021

@author: shreyanskothari
"""

import os
os.chdir("/Users/shreyanskothari/Desktop/Natural Language Processing/Practice_Review")
from utils.my_funcs import *
import pickle

my_pd = pickle.load(open("my_pd.pkl", "rb"))


my_pd['jaccard_similarity'] = my_pd.body.apply(lambda x: jaccard_fun(x, "Fishing is fun"))

emb_data = pickle.load(open("embeddings_df.pkl", "rb"))

cos_matrix = cosine_fun(emb_data, emb_data, my_pd.label)


emb_data_domain = pickle.load(open("embeddings_df_domain.pkl", "rb"))

cosine_matrix_domain = cosine_fun(emb_data_domain, emb_data_domain, my_pd.label)

"""Count Vectorization:
    - looks at all the unique words based off the given corpuses
    - gives the frequrncy of occurance of each of those unique words in all 
    the corpuses, on a corpus-by-corpus basis
    - Columns represent unique tokens
    - each rows represents a single corpus
    - ngram_range: argument to set the number of ngrams that should be analyzed 
    in the count vectorizer. Ability to utilize bigrams, trigrams, etc. 
    Don't have to focus only on unigrams.
    ngram_range = (1,2): include both unigrams and bigrams
    ngram_range(1, 3): includes unigrams, bigrams, and trigrams
    
    """
    
path_o = "/Users/shreyanskothari/Desktop/Natural Language Processing/Practice_Review/"
my_vec_data = my_vec_fun(my_pd.body, 1, 1, path_o)
    

"""Term Frequency - Inverse Document Frequency (TF-IDF):
    - Measure of originiality of a word 
    - Compares the number of times a word appears in a document with the 
    number of documents the word appears in.
    
    TF-IDF = TF(t,d) X IDF(t)
    - TF(t,d): number of times a term, t, appears in a document, d
    - IDF(t): Document frequency of the term t
   
          
    TF-IDF = TF(t,d) * IDF(d)
    
    
    The formula that is used to compute the tf-idf for a term t of a document 
    d in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is 
    computed as idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False), where n 
    is the total number of documents in the document set and df(t) is the 
    document frequency of t; the document frequency is the number of documents 
    in the document set that contain the term t. 
    tf is the number of times t appears in a document, divdied by the total 
    number of words in the document.
    """
    
my_tf_idf_data = my_tf_idf_fun(my_pd.body, 1, 1, path_o)
    
