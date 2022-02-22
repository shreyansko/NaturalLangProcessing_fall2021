#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:26:14 2021

@author: shreyanskothari
"""

from utils.my_functions import *
import nltk
from nltk.corpus import stopwords
sw = set(stopwords.words('English'))

the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/corpuses"

my_pd = file_seeker2(the_path, "check_words")


#create a new column on my_pd called "body_sw" which will be the result of 
#removing stop words from the body column

def nosw(filein):
    import nltk
    from nltk.corpus import stopwords
    sw = set(stopwords.words('English'))
    my_text = [word for word in filein.split() if word not in sw]
    my_text =' '.join(my_text)
    return my_text
    
#in pandas, you can run the apply function and it runs a specified function on all values of the columb
my_pd['body_sw'] =  my_pd.body.apply(nosw)

dictionary = nltk.corpus.words.words('en')

#function that removes words not in the english dictionary
def check_dictionary(var):
    import nltk
    dictionary = nltk.corpus.words.words('en')
    my_text = [word for word in var.split() if word in dictionary]
    my_text = ' '.join(my_text)
    return my_text

my_pd['sw_dictionary'] = my_pd.body.apply(check_dictionary)

#you can save any object as a pickle object and load it back
#API that lets you save any file as a pickle object, which shows up in our working directory
#
import pickle
pickle.dump(my_pd, open( "my_pd.pkl", "wb")) #wb = write binary
test = pickle.load(open("my_pd.pkl", "rb")) #rb = read binary

#removes stem from the word
from nltk.stem.porter import *
stemmer = PorterStemmer()

stemmer.stem("running")

#Create a new column "sw_stem" that stems every word (removes suffix and prefix including 'ing')
test['sw_stem'] = test.body.apply(stem_fun)

def stem_fun(filein):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    my_text = [stemmer.stem(word) for word in filein.split()]
    my_text = ' '.join(my_text)
    return my_text



