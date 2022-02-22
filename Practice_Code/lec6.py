#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:52:21 2021

@author: shreyanskothari
"""
from utils.my_funcs import *
import pickle


the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/corpuses"

my_pd = pickle.load(open("my_pd.pkl", "rb"))
#pickle.dump(my_pd, open("my_pd.pkl", "wb"))

my_pd['jaccard_similarity'] = my_pd.body.apply(lambda x: jaccard_fun(x, "Fishing is fun"))

"""Jaccardian Similarity:
    Shortcoming: doesn't have sensitivity to sizes of the two sets
    if set A has 50 elements and set B has 50 elements, 
    the jaccard similarity is a certain number. But, if you increase 
    set A by 10 and decrease B by 10, they would still have a 100 elements 
    together and the jaccardian similarity would still be similar
    
    Distributional hypothesis: the link between similarity in how words are 
    distributed and the similarity in what they mean.
    
    Embeddings: linguist hypothesis by representations of the meanings of words.
    Embedding is a term used for the representation of words for text analysis, 
    typically in the form of a real-valued vector (each ement of the vector is a 
    real number) that encodes the meaning of the word such that the words that are
    closer in the vector space are expected to be similar in meaning.
    Embeddings depend on word similarity
    Example:
        Vector[Queen] ≈ Vector[King] - Vector[Man] + Vector[woman]
    
    
    
    Lemma: Also called Citation Forms. The root word (stem) without any action 
    verbs applied to it. Each lemma can have various meanings. Example- the lemma
    'Mouse' can mean a rodent or a device used with computers. 
    Senses: Senses are the concepts/meanings of the words/lemmas. 
    Example:
        
        Lemma : Mouse
        Senses: 1. rodent
                2. device used with computers
    
    Synonymys: Two words are synonyms when they have the same propositional meaning.
    They are substitutable without changing the truth conditions of the sentence. 
    
    Principle of Contrast: difference in linguistic form is always associated with 
    some difference in meaning. 
    Difference in form --> Difference in meaning
    
    Similarity: Knowing how similar two words are allows us to measure how similar
    the meaning of two sentences is. 
    
    Relatedness: meaning of two words can be related without being similar. Example:
        coffee and cup.
    
    Semantic field: a set of words that cover a particular semantic domain and bear
    structured relations with each other. 
    
    
    word2vec: short, dense vectors. 
    - Representation created by training a classifier to predict whether a word is
    likely to appear nearby
    - computed with meaning representations instead of string representations
    - Preserves the relationship between words 
    - Two training algorithms that make up the wordvec algorithm:
        - Continuous bag of words (CBOW): 
            - Context to predict main target word
            - Goes from many words to predicting one main target word
        - Skip-gram: (opposite of CBOW)
            - from one word, capture semantics for a bunch of different words
            - all those words will be in someway related to that one word
    
    """