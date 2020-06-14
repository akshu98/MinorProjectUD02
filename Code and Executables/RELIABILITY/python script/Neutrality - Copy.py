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
    sid_obj = SentimentIntensityAnalyzer()  
    sentiment_dict = sid_obj.polarity_scores(text)    
    return sentiment_dict['compound']


# In[6]:


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


# In[7]:


sentimentcaa0=caacluster0['sentiment']
sentimentcaa1=caacluster1['sentiment']
sentimentcaa2=caacluster2['sentiment']
sentimentcovid0=covidcluster0['sentiment']
sentimentcovid1=covidcluster1['sentiment']
sentimentcovid2=covidcluster2['sentiment']


# In[8]:


def neutrality_score(articleScoreDict):
    articleValues=list(articleScoreDict)
    neutrality=[]
  #  caacluster0["neutrality"]=""
   # print(articleValues)
    scaler= MinMaxScaler()
    scaler.fit(np.asarray(articleValues))
    a=scaler.transform(articleValues)
    
    mean=np.mean(a)
   # print(mean)
    for i in a:
        diff=(abs(mean-i)**2)
        neutrality.append(diff)
    caacluster1['neutrality']=neutrality
    
    caacluster1.to_csv("../data/caa_nrc_cluster_1.csv") 
   # caacluster1.to_csv("caa_nrc_cluster_1.csv") 
   # caacluster2.to_csv("caa_nrc_cluster_2.csv")
   # covidcluster2.to_csv("covid_cluster_2.csv")


# In[9]:


neutrality_score(sentimentcaa1)
#computePopulationBalanceScoreHistoMean(sentimentcaa1)
#computePopulationBalanceScoreHistoMean(sentimentcaa2)
#computePopulationBalanceScoreHistoMean(sentimentcovid0)
#computePopulationBalanceScoreHistoMean(sentimentcovid1)
#computePopulationBalanceScoreHistoMean(sentimentcovid2)

