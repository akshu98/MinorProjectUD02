#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet 
from collections import Counter,OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import enchant
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newspaper import Article
import statistics
from sklearn.preprocessing import MinMaxScaler


# In[2]:


caacluster0=pd.DataFrame(pd.read_csv("../data/caa_nrc_cluster_0.csv"))
caacluster1=pd.DataFrame(pd.read_csv("../data/caa_nrc_cluster_1.csv"))
caacluster2=pd.DataFrame(pd.read_csv("../data/caa_nrc_cluster_2.csv"))
covidcluster0=pd.DataFrame(pd.read_csv("../data/covid_cluster_0.csv"))
covidcluster1=pd.DataFrame(pd.read_csv("../data/covid_cluster_1.csv"))
covidcluster2=pd.DataFrame(pd.read_csv("../data/covid_cluster_2.csv"))


# In[3]:


caacluster0['Text']=caacluster0['Text'].astype(str)
caacluster1['Text']=caacluster1['Text'].astype(str)
caacluster2['Text']=caacluster2['Text'].astype(str)
covidcluster0['Text']=covidcluster0['Text'].astype(str)
covidcluster1['Text']=covidcluster1['Text'].astype(str)
covidcluster2['Text']=covidcluster2['Text'].astype(str)



# In[4]:


caaArticles0=[]
caaArticles1=[]
caaArticles2=[]
covidArticles0=[]
covidArticles1=[]
covidArticles2=[]
for i in caacluster0.index:
    text=caacluster0['Text'][i]
    caaArticles0.append(text)
for i in caacluster1.index:
    text=caacluster1['Text'][i]
    caaArticles1.append(text)
for i in caacluster2.index:
    text=caacluster2['Text'][i]
    caaArticles2.append(text)
for i in covidcluster0.index:
    text=covidcluster0['Text'][i]
    covidArticles0.append(text)
for i in covidcluster1.index:
    text=covidcluster1['Text'][i]
    covidArticles1.append(text)
for i in covidcluster2.index:
    text=covidcluster2['Text'][i]
    covidArticles2.append(text)


# In[5]:

def sentiment_scores(text): 
    #print(index)
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(text) 
    
#     print("text Overall Rated As", end = " ") 
  
#     # decide sentiment as positive, negative and neutral 
#     if sentiment_dict['compound'] >= 0.05 : 
#         print("Positive",index) 
  
#     elif sentiment_dict['compound'] <= - 0.05 : 
#         print("Negative",index) 
  
#     else : 
#         print("Neutral",index) 
#     print("\n")
    
    return sentiment_dict['compound']




class SentimentAnalyser():

    scaleMin=-1.
    scaleMax=1.

    # Initializer / Instance attributes
    def __init__(self, library):
        if library=='vader':
            self.analyser=NLTKVaderSentimentAnalyser()


# In[7]:


caacluster0["sentiment"]=""
caacluster1["sentiment"]=""
caacluster2["sentiment"]=""
covidcluster0["sentiment"]=""
covidcluster1["sentiment"]=""
covidcluster2["sentiment"]=""

for i in range(len(caaArticles0)):
    article_title = caacluster0['Title'].iloc[i]
    article_text = caacluster0['Text'].iloc[i]    
    caacluster0['sentiment'][i] = sentiment_scores(article_text)
for i in range(len(caaArticles1)):
    article_title = caacluster1['Title'].iloc[i]
    article_text = caacluster1['Text'].iloc[i]    
    caacluster1['sentiment'][i] = sentiment_scores(article_text)
for i in range(len(caaArticles2)):
    article_title = caacluster2['Title'].iloc[i]
    article_text = caacluster2['Text'].iloc[i]    
    caacluster2['sentiment'][i] = sentiment_scores(article_text)
for i in range(len(covidArticles0)):
    article_title = covidcluster0['Title'].iloc[i]
    article_text = covidcluster0['Text'].iloc[i]    
    covidcluster0['sentiment'][i] = sentiment_scores(article_text)
for i in range(len(covidArticles1)):
    article_title =covidcluster1['Title'].iloc[i]
    article_text = covidcluster1['Text'].iloc[i]    
    covidcluster1['sentiment'][i] = sentiment_scores(article_text)
for i in range(len(covidArticles2)):
    article_title = covidcluster2['Title'].iloc[i]
    article_text = covidcluster2['Text'].iloc[i]    
    covidcluster2['sentiment'][i] = sentiment_scores(article_text)


sentimentcaa0=list(caacluster0['sentiment'])
sentimentcaa1=list(caacluster1['sentiment'])
sentimentcaa2=list(caacluster2['sentiment'])
sentimentcovid0=list(covidcluster0['sentiment'])
sentimentcovid1=list(covidcluster1['sentiment'])
sentimentcovid2=list(covidcluster2['sentiment'])



def neutrality_score(articleScoreDict):
    articleValues=articleScoreDict
    neutrality=[]
    a = np.asarray(articleValues)
    s=np.reshape(a,(-1,1))
    scaler = MinMaxScaler()
    scaler.fit(s)
    a=scaler.transform(s)
   
    mean=np.mean(a)
    for i in a:
        diff=(abs(mean-i)**2)
        neutrality.append(diff)
    # print("Neutrality ",neutrality)
    return neutrality



def neutrality_score_finder(cluster_number,flag,article):
    print("neutrality_score_finder called")

    if flag ==1:

        if cluster_number==0:
            sent_10 = []
            val = sentiment_scores(article)
            sentimentcaa0.append(val)
            n_10 = neutrality_score(sentimentcaa0)
            return n_10[len(n_10)-1][0]

        elif cluster_number==1:
            sent_10 = []
            val = sentiment_scores(article)
            sentimentcaa1.append(val)
            n_10 = neutrality_score(sentimentcaa1)
            return n_10[len(n_10)-1][0]

        else:

            sent_10 = []
            val = sentiment_scores(article)
            sentimentcaa2.append(val)
            n_10 = neutrality_score(sentimentcaa2)
            return n_10[len(n_10)-1][0]

    else:
        if cluster_number==0:
            sent_10 = []
            val = sentiment_scores(article)
            sentimentcovid0.append(val)
            n_10 = neutrality_score(sentimentcovid0)
            return n_10[len(n_10)-1][0]

        elif cluster_number==1:
            sent_10 = []
            val = sentiment_scores(article)
            sentimentcovid1.append(val)
            n_10 = neutrality_score(sentimentcovid1)
            return n_10[len(n_10)-1][0]

        else:
            caaArticles0.append(article)
            sent_10 = []
            val = sentiment_scores(article)
            sentimentcovid2.append(val)
            n_10 = neutrality_score(sentimentcovid2)
            return n_10[len(n_10)-1][0]


    






