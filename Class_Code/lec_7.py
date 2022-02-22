# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:35:44 2021

@author: pathouli
"""

from utils.my_functions import *
import pickle

the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/Code"

#my_pd = file_seeker(the_path, False)

#my_pd["body_sw"] = my_pd.body.apply(rem_sw)
#pickle.dump(my_pd, open("my_pd.pkl", "wb"))

#my_pd["sw_stem"] = my_pd.body_sw.apply(stem_fun)
#pickle.dump(my_pd, open("my_pd.pkl", "wb"))

my_pd = pickle.load(open("my_pd.pkl", "rb"))

my_pd["jaccard_similarity"] = my_pd.body_sw.apply(
    lambda x: jaccard_fun(x, "machine learning is fun"))

the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/Code"
the_path_in = "/Users/shreyanskothari/Desktop/Natural Language Processing/Code/word2vec-GoogleNews-vectors-master.zip"
#tmp_data =  extract_embeddings_pre(
#    my_pd.body_sw, 300, the_path, the_path_in + "GoogleNews-vectors-negative300.bin.gz")


emb_data = pickle.load(open( "/Users/shreyanskothari/Desktop/Natural Language Processing/Code/embeddings_df.pkl", "rb"))

cos_matrix = cosine_fun(emb_data, emb_data, my_pd.label)

# emb_data_domain, word_dict =  extract_embeddings_domain(
#     my_pd.sw_stem, 300, the_path)

emb_data_domain = pickle.load(
    open(the_path + "embeddings_df_domain.pkl", "rb"))

cos_matrix_domain = cosine_fun(emb_data_domain, emb_data_domain, my_pd.label)

my_vec_data = my_vec_fun(my_pd.sw_stem, 1, 3, the_path)

my_tf_idf_data = my_tf_idf_fun(my_pd.sw_stem, 1, 3, the_path)

model, metrics, my_preds = my_model_fun(my_vec_data, my_pd.label, the_path)



df_in = my_vec_data


#return my_model, model_metrics, the_preds



df_in = my_vec_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import pickle
my_model = RandomForestClassifier(max_depth=10, random_state=123)
#my_model = GaussianNB()
#80/20 train,test,split
X_train, X_test, y_train, y_test = train_test_split(
    df_in, my_pd.label, test_size=0.20, random_state=123)
my_model.fit(X_train, y_train)
pickle.dump(my_model, open(the_path + "my_model.pkl", "wb"))
y_pred = my_model.predict(X_test)
model_metrics = pd.DataFrame(precision_recall_fscore_support(
y_test, y_pred, average='weighted'))
model_metrics.index = ["precision", "recall", "fscore", "none"]
#function 2 prediction
the_preds = pd.DataFrame(my_model.predict_proba(X_test))
the_preds.columns = my_model.classes_


# def get_embed(path_in, file_name_in, var_in):
#     import numpy as np
#     embed_model = open_pickle(path_in, file_name_in)
#     tmp = var_in.split()
#     def get_score(var):
#         try:
#             tmp_arr = list()
#             for word in tmp:
#                 tmp_arr.append(list(embed_model.wv[word]))
#         except:
#             pass
#         return np.mean(np.array(tmp_arr), axis=0)
#     tmp_out = get_score(var_in).reshape(1, -1)
#     return tmp_out