#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:16:15 2021

@author: shreyanskothari
"""

"""Lecture 4 functions"""

### File Seeker Function
def file_seeker(path_in, sw):
    import os
    import pandas as pd
    from utils.my_funcs import clean_text
    import nltk
    dictionary = nltk.corpus.words.words("en")
    df = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
         dir_name = dirName.split('/')[-1::][0]
         try:
            for word in fileList:
                 txt = open(dirName+'/'+word, 'r')
                 txt_df = txt.read()
                 txt.close()
                 clean = clean_text(txt_df)
                 if sw == "check_words": #Added in lecture 5 after check_dictionary function
                     tmp = [word for word in clean.split() if word in dictionary]
                     tmp = ' '.join(tmp)
                 df = df.append({"label": dir_name, 
                                "body": clean},
                               ignore_index = True)
         except Exception as e:
            print(e)
            pass #even if there is an error, the loop keeps going on
    return df


### Clean Text function
def clean_text(txt_in):
    import re
    clean = re.sub('[^A-Za-z0-9]+', " ", txt_in).strip()
    return clean.lower()


"""Lecture 5 functions"""

### Function to remove stopwords
def rem_sw(txt_in):
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    sw = set(stopwords.words("English"))
    tmp = txt_in.split()
    tmp = [word for word in tmp if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

### Function to only include words in the dictionary
def check_dictionary(txt_in):
    import nltk
    dictionary = nltk.corpus.words.words("en")
    tmp = [word for word in txt_in.split() if word in dictionary]
    tmp = ' '.join(tmp)
    return tmp

###Function for stemming words (get stem words, aka without 'ing, 's', etc.) in text files
def stem_fun(txt):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tmp = txt.split()
    tmp = [stemmer.stem(word) for word in tmp]
    tmp = ' '.join(tmp)
    return tmp


"""Lecture 6 functions"""

###Function for measuring jaccardian distance: intersection/union
def jaccard_fun(txt1, txt2):
     tmp1 = set(txt1.split())
     tmp2 = set(txt2.split())
     inters = tmp1.intersection(tmp2)
     unio = tmp1.union(tmp2)
     jaccardian =float(len(inters))/float(len(unio))
     return jaccardian
 
    
"""Lecture 7 functions"""

###Function to give a cosine similarity matrix
def cosine_fun(df1, df2, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_matrix_t = pd.DataFrame(cosine_similarity(df1, df2))
    cos_matrix_t.index = label_in
    cos_matrix_t.columns = label_in
    return cos_matrix_t

###Function to create a count vectorizer matrix. Also saves the vectorizer model
### and resulting dataframe as pickle objects
def my_vec_fun(df, m, n, path_o):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import pickle
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec = pd.DataFrame(vectorizer.fit_transform(df).toarray())
    my_vec.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "vectorizer.pkl", "wb"))
    pickle.dump(my_vec, open(path_o + "my_vec_data.pkl", "wb"))
    return my_vec

###Function to create a tf-idf vectorizer matrix. Also saves the vectorizer model
### and resulting dataframe as pickle objects
def my_tf_idf_fun(df, m, n, path_o):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    tf_idf = TfidfVectorizer(ngram_range=(m,n))
    my_tfidf = pd.DataFrame(tf_idf.fit_transform(df).toarray())
    my_tfidf.columns = tf_idf.vocabulary_
    pickle.dump(tf_idf, open(path_o + "tf_idf.pkl", "wb"))
    pickle.dump(my_tfidf, open(path_o + "my_tf_idf_data.pkl", "wb"))
    return my_tfidf