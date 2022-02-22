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

#create a new function that compares two corpuses and outputs the 
#jaccardian similarity score

def jaccard_fun(var_a, var_b):
    #DOCUMENT SIMILARITY
    doc_a_set = set(var_a.split())
    doc_b_set = set(var_b.split())
    j_d = float(len(doc_a_set.intersection(
        doc_b_set)))/ float(len(doc_a_set.union(doc_b_set)))
    return j_d


#print (jaccard_fun("machine learning is fun", "machine learning"))

my_pd["jaccard_similarity"] = my_pd.body.apply(
    lambda x: jaccard_fun(x, "fishing is fun"))


the_path = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/GR5067/2021_fall/code/"
the_path_in = "C:/Users/pathouli/myStuff/academia/columbia/socialSciences/word2vec/"
tmp_data =  extract_embeddings_pre(my_pd.body, 300, the_path, the_path_in + "GoogleNews-vectors-negative300.bin.gz")





def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    from gensim.models import Word2Vec
    import pandas as pd
    from gensim.models import KeyedVectors
    import pickle
    my_model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    my_model = Word2Vec(df_in.str.split(),
                        min_count=1, vector_size=num_vec_in)#size=300)
    #word_dict = my_model.wv.key_to_index
    #my_model.most_similar("calculus")
    #my_model.similarity("trout", "fish")
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(my_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pkl", "wb" ))
    return tmp_data#, word_dict

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    wrd_dict = model.wv.key_to_index
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(base_path + "embeddings_domain.pkl")
    pickle.dump(model, open(path_in + "embeddings_domain_model.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pkl", "wb" ))
    return tmp_data, wrd_dict

def get_embed(path_in, file_name_in, var_in):
    import numpy as np
    embed_model = open_pickle(path_in, file_name_in)
    tmp = var_in.split()
    def get_score(var):
        try:
            tmp_arr = list()
            for word in tmp:
                tmp_arr.append(list(embed_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = get_score(var_in).reshape(1, -1)
    return tmp_out