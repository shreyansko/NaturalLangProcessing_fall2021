#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:01:26 2021

@author: shreyanskothari
"""
 
import os
os.getcwd()

os.chdir('/Users/shreyanskothari/Desktop/Natural Language Processing/Assignments/Assignment1')

#opening hw.pkl
import pickle
my_pd = pickle.load(open("hw1.pkl", "rb"))

import pandas as pd

#Creating a dictionary called corpus_analytics
corpus_analytics = dict()

body_basic = pd.DataFrame()
lower = pd.DataFrame()
sw =  pd.DataFrame()

################################# token #######################################

#Creating a column in body_basic, lower, and sw called token where each row 
# represents a unique token(word)

#body_list.token
body_list = [word for word in my_pd.body_basic.str.split()]
body_list = [item for sublist in body_list for item in sublist]
body_set = set(body_list)
body_basic["token"] = [row for row in body_set]

#lower.token
lower_list = [word for word in my_pd.lower.str.split()]
lower_list = [item for sublist in lower_list for item in sublist]
lower_set = set(lower_list)
lower["token"] = [row for row in lower_set]

#sw.token
sw_list = [word for word in my_pd.sw.str.split()]
sw_list = [item for sublist in sw_list for item in sublist]
sw_set = set(sw_list)
sw["token"] = [row for row in sw_set]

############################## raw_frequency ##################################
#raw_frequency: total number of times the specific token shows


##Adding raw_frequency to body_basic dataframe 

#Creating empty dataframe
x = pd.DataFrame()

#Adding all words from all texts from body_basic to x 
x["words"] = [row for row in body_list]

#Creating new variable y which contains value counts for all unique words in x
y = x.value_counts()

#resetting the index for ease in joining dataframes using inner-join by token
y = y.reset_index()
y = y.rename(columns = {"words":"token", 0:"raw_frequency"})
body_basic = pd.DataFrame.merge(body_basic, y, on = 'token')


##Adding raw_frequency to lower dataframe 
x1 = pd.DataFrame()
x1["words"] = [row for row in lower_list]
y1 = x1.value_counts()
y1 = y1.reset_index()
y1 = y1.rename(columns = {"words":"token", 0:"raw_frequency"})
lower = pd.DataFrame.merge(lower, y1, on = 'token')


## Adding raw_frequency to sw dataframe
x2 = pd.DataFrame()
x2["words"] = [row for row in sw_list]
y2 = x2.value_counts()
y2 = y2.reset_index()
y2 = y2.rename(columns = {"words":"token", 0:"raw_frequency"})
sw = pd.DataFrame.merge(sw, y2, on = 'token')

############################## probability(%) ##################################
#out of the count of all tokens (non-unique) what % of times does this word show up?

body_basic["probability (%)"] = body_basic["raw_frequency"]/len(body_list) * 100
lower["probability (%)"] = lower["raw_frequency"]/len(lower_list) * 100 
sw["probability (%)"] = sw["raw_frequency"]/len(sw_list) * 100 

################ Sorting by probability in descending order ###################
body_basic = body_basic.sort_values(by = "probability (%)", ascending= False)
lower = lower.sort_values(by = "probability (%)", ascending= False)
sw = sw.sort_values(by = "probability (%)", ascending= False)

################### guessing corpus topic based on head(10) ###################
body_basic.head(10)
"""
body_basic:
Based on the 10 most frequent words in body_basic, it is almost impossible to guess
the topic of the corpus. All of these seem to be stopwords and need to be removed
in order to actually understand the topic of the corpus. 

"""

lower.head(10)
"""
lower:
Based on the 10 most frequent words in lower, it is still very difficult to
guess the topic of the corpus. These are mostly still stopwords. However, it is 
relatively easier to guess for lower than it is for body_basic. Looking at the 
top ten probabilities, we can see that entomology is one of them. Entomology is
the study of insects so the corpus is likely about insects and/or their study.
    
"""

sw.head(10)
"""
sw:
Based on the 10 most frequent words in sw, it is very easy to guess the topic of
this corpus. The corpus is about entomology, research, and collection of different
species of insects. It seems like it could be related to a college / university 
and their entomology department where people research new species of insects.
    
"""

corpus_analytics["body_basic"] = body_basic
corpus_analytics["lower"] = lower
corpus_analytics["sw"] = sw

