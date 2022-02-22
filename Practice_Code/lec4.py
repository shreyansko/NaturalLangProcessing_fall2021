#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:53:56 2021

@author: shreyanskothari
"""


import os
os.chdir("/Users/shreyanskothari/Desktop/Natural Language Processing/Practice_Review")


from utils.my_funcs import *

the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/corpuses"
my_pd = file_seeker(the_path)

#Removing first row for Label = corpuses
my_pd = my_pd.drop(index = 0)

my_pd.label.value_counts()
#could also do
#my_pd.groupby('Label').count()

