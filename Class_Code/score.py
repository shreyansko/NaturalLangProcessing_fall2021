# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:40:29 2021

@author: pathouli
"""

def score_text(sample_text_i, the_path_i):
    #Classify arbitrary text, need to run ALL same pre-processing and 
    #transformation steps as what was applied to the training set
    #Step 1 call; up clean_text function
    import numpy as np
    from utils.my_functions import clean_text, stem_fun
    import pickle 
    clean_text = clean_text(sample_text_i)
    #step 2 call up stem_fun
    clean_text = stem_fun(clean_text)
    #call up the appropriate transformation algo and transform our new data
    vec_model = pickle.load(open(the_path_i + "vectorizer.pkl", "rb"))
    clean_text_vec = vec_model.transform([clean_text]).toarray()
    my_model = pickle.load(open(the_path_i + "my_model.pkl", "rb"))
    the_pred = my_model.predict(clean_text_vec)
    the_pred_proba = np.max(np.array(my_model.predict_proba(clean_text_vec)))
    print ("sample_text predicted as", the_pred[0],
           "with likelihood of", the_pred_proba)
    return the_pred, the_pred_proba

the_path = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/GR5067/2021_fall/code/"
sample_text = "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."

my_pred, my_pred_score = score_text(sample_text, the_path)

