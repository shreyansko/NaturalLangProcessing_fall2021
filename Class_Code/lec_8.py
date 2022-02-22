# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:35:44 2021

@author: pathouli
"""

from utils.my_functions import *
import pickle

the_path = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/GR5067/2021_fall/data/"

#my_pd = file_seeker(the_path, False)

#my_pd["body_sw"] = my_pd.body.apply(rem_sw)
#pickle.dump(my_pd, open("my_pd.pkl", "wb"))

#my_pd["sw_stem"] = my_pd.body_sw.apply(stem_fun)
#pickle.dump(my_pd, open("my_pd.pkl", "wb"))

my_pd = pickle.load(open("my_pd.pkl", "rb"))

my_pd["jaccard_similarity"] = my_pd.sw_stem.apply(
    lambda x: jaccard_fun(x, "machine learning is fun"))

the_path = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/GR5067/2021_fall/code/"
the_path_in = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/word2vec/"
#tmp_data =  extract_embeddings_pre(
#    my_pd.body_sw, 300, the_path, the_path_in + "GoogleNews-vectors-negative300.bin.gz")

import pickle
emb_data = pickle.load(open(the_path + "embeddings_df.pkl", "rb"))

cos_matrix = cosine_fun(emb_data, emb_data, my_pd.label)

# emb_data_domain, word_dict =  extract_embeddings_domain(
#     my_pd.sw_stem, 300, the_path)

emb_data_domain = pickle.load(
    open(the_path + "embeddings_df_domain.pkl", "rb"))

#various transformation algorithms
cos_matrix_domain = cosine_fun(emb_data_domain, emb_data_domain, my_pd.label)
my_vec_data = my_vec_fun(my_pd.sw_stem, 1, 1, the_path)
my_tf_idf_data = my_tf_idf_fun(my_pd.sw_stem, 1, 1, the_path)
my_pca_vec_i = my_pca_fun(my_tf_idf_data, 0.95, the_path)

model, metrics, my_preds = my_model_fun_grid(
    my_vec_data, my_pd.label, the_path)


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