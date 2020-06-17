#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import wordnet
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

import re


# In[2]:


articles_df = pd.read_csv("../data/parsed_cleaned.csv")
articles_df['Cleaned_text'] = articles_df['Cleaned_text'].astype(str)


# In[3]:


with open('../data/bias_word_list_01_2018.txt') as f:
    words = f.readlines()
    
words_to_check = []
for i in words:
    words_to_check.append(i.strip('\n'))


# In[4]:


# print(words_to_check)


# In[5]:


# synonyms = []
# antonyms = []

# for word in LOW:
#     for syn in wordnet.synsets(word):
#         for l in syn.lemmas():
#             synonyms.append(l.name())
#             if l.antonyms():
#                 antonyms.append(l.antonyms()[0].name())

# synonyms = list(set(synonyms))
# antonyms = list(set(antonyms))

# print(len(synonyms))
# print(len(antonyms))

# words_to_check = list(set(synonyms+LOW))


# for word in words_to_check:
#     word = lemmatizer.lemmatize(word)
    


# In[6]:


# len(words_to_check)


# In[7]:


articles_df["strongP"]=""
scores = []
for article in articles_df["Cleaned_text"]:
    c=0
    
    for word in set(article.split(' ')):
        
        if(word in words_to_check):
            c=c+1    
    try:
        scores.append(float(c)/len(set(article.split(' '))))
        
    except Exception:
        scores.append(0.0)

articles_df["strongP"]=scores


# In[8]:


# print(max(articles_df['strongP']))
# print(min(articles_df['strongP']))


# In[9]:


for i in range(len(articles_df)):
    if articles_df['strongP'][i]>=0 and articles_df['strongP'][i]<=0.05:
        articles_df['strongP'][i]=3
    elif articles_df['strongP'][i]>0.05 and articles_df['strongP'][i]<=0.1:
        articles_df['strongP'][i]=2
    else:
        articles_df['strongP'][i]=1


# In[10]:


articles_df.sort_values(by='strongP')


# In[11]:


articles_df.to_csv('../data/StrongWordsAnalysis.csv',index=None)








def strong_words_score(article):
    c=0
    print("length",len(words_to_check))
    for word in set(article.split(' ')):

        if(word in words_to_check):
            c=c+1   
             
    try:
        score = float(c)/len(set(article.split(' ')))
        # print("c",c)
        # print(len(set(article.split(' '))))
    except Exception:
        score = 0.0
    #print("score",score)
    print("article",article)
    if score>=0 and score<=0.05:
        score_label=3
    elif score>0.05 and score<=0.1:
        score_label=2
    else:
        score_label=1

    return score_label
