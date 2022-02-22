# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:59:46 2021

@author: pathouli
"""

import pickle
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np

def fetch_bi_grams(var):
    sentence_stream = np.array(var)
    bigram = Phrases(sentence_stream, min_count=5, threshold=10, delimiter=",")
    trigram = Phrases(bigram[sentence_stream], min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_grams.append(bigram_phraser[sent])
        tri_grams.append(trigram_phraser[sent])
    return bi_grams, tri_grams

if __name__ == '__main__':
    the_path = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/GR5067/2021_fall/code/"
    
    the_data_t = pickle.load(open(the_path + "data_t.pkl", "rb")).str.split() #unigram
    
    bi, tri = fetch_bi_grams(the_data_t)

    the_data = the_data_t

    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    
    corpus = [id2word.doc2bow(text) for text in the_data]
    
    n_topics = 5
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=n_topics, id2word=id2word, iterations=50, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
        
    #compute Coherence Score using c_v
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                         dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    c_scores = list()
    for word in range(1, n_topics):
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=id2word, iterations=10, passes=5)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())
    
    x = range(1, n_topics)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
