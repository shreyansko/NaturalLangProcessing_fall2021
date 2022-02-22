#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:52:15 2020
@author: shreyko

For my Natural Language Processing class, I created a real-time reddit post
analyzer that classifies whether someone is more democratic or republican, based
on the text in their post. I utilized a pre-trained ML model for this assignment.

Please note that you would need a username and password, and a client_id and client_secret
to run this model. I used a reddit API which requires the user to create an app on the
platform; we need use our login credentials to pull data from the website.
"""

import praw
import pandas as pd
import pytz
from datetime import datetime

subreddit_channel = 'politics'

reddit = praw.Reddit(
     client_id="_____",
     client_secret="_____",
     user_agent="testscript by u/fakebot3",
     username="_____",
     password="_____",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    from utils.my_functions import clean_text
    from utils.my_functions import rem_sw
    from utils.my_functions import stem_fun
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time,
                'body_clean': clean_text(var_in.body),
                'body_sw': rem_sw(clean_text(var_in.body)),
                'body_stem': stem_fun(rem_sw(clean_text(var_in.body)))}
    import pickle
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    my_pca = pickle.load(open("pca.pkl", "rb"))
    model = pickle.load(open("my_model.pkl", "rb"))
    
    my_vec = vectorizer.transform([tmp_dict['body_stem']])
    my_pca_vec = my_pca.transform(my_vec.toarray())
    the_preds = pd.DataFrame(model.predict_proba(my_pca_vec))
    the_preds.columns = model.classes_
    #return tmp_dict
    return var_in.body, the_preds
    
for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    print(tmp_df, '\n \n')
