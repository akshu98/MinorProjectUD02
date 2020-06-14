#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[1]:


from __future__ import print_function
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant
from sklearn.cluster import KMeans
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.stem import WordNetLemmatizer 

import enchant

import os
import time

lemmatizer = WordNetLemmatizer()


# # Reading the news articles into a dataframe

# In[2]:


df =pd.read_csv("../data/parsed_cleaned.csv")
df['Text'] = df['Text'].astype(str)

caa_df = df[df['Topic Name']=='caa-nrc']
covid_df = df[df['Topic Name']=='coronavirus']


# In[7]:


articles_caa = caa_df['Cleaned_text']
articles_covid = covid_df['Cleaned_text']
vectorizer_caa = TfidfVectorizer()
vectorizer_covid = TfidfVectorizer()
tfidf_matrix_caa = vectorizer_caa.fit_transform(articles_caa.values.astype('U'))
tfidf_matrix_covid = vectorizer_covid.fit_transform(articles_covid.values.astype('U'))
# print(tfidf_matrix_caa.shape)
# print(tfidf_matrix_covid.shape)
terms_caa = vectorizer_caa.get_feature_names()
terms_covid = vectorizer_covid.get_feature_names()


# In[8]:


num_clusters = 3
km_caa = KMeans(n_clusters=num_clusters,max_iter=100)
km_covid = KMeans(n_clusters=num_clusters,max_iter=100)
km_caa.fit(tfidf_matrix_caa)
km_covid.fit(tfidf_matrix_covid)
clusters_caa = km_caa.labels_.tolist()
clusters_covid = km_covid.labels_.tolist()


# In[9]:


caa_df['Clusters']=clusters_caa
covid_df['Clusters']=clusters_covid


# In[10]:


# print("Top terms per cluster CAA-NRC:")
# order_centroids = km_caa.cluster_centers_.argsort()[:, ::-1]
# for i in range(num_clusters):
#     print("Cluster %d:" % i)
#     for ind in order_centroids[i, :10]:
#         print(terms_caa[ind])
#     print("----------------")
    
    
# print("\n\nTop terms per cluster COVID-19:")
# order_centroids = km_covid.cluster_centers_.argsort()[:, ::-1]
# for i in range(num_clusters):
#     print("Cluster %d:" % i)
#     for ind in order_centroids[i, :10]:
#         print(terms_covid[ind])
#     print("----------------")


# In[11]:


for i in range(num_clusters):
    articles=[]
    file_name = "../data/caa_nrc_cluster_"+str(i)+".csv"
    for j in caa_df.index:
        
        if caa_df['Clusters'][j]==i:
            title = caa_df['Title'][j]
            title = re.sub('[^\\x00-\\x7F]+','', title)
            article = caa_df['Cleaned_text'][j]
            articles.append([title,article])
    temp = pd.DataFrame(articles,columns=['Title','Text'])
    temp.to_csv(file_name,index=None)
    articles.clear()
    


# In[12]:


for i in range(num_clusters):
    articles=[]
    file_name = "../data/covid_cluster_"+str(i)+".csv"
    for j in covid_df.index:
        
        if covid_df['Clusters'][j]==i:
            title = covid_df['Title'][j]
            title = re.sub('[^\\x00-\\x7F]+','', title)
            article = covid_df['Cleaned_text'][j]
            articles.append([title,article])
    temp = pd.DataFrame(articles,columns=['Title','Text'])
    temp.to_csv(file_name,index=None)
    articles.clear()
    


def cluster_new_article(article,topic_name):
    print("cluster_new_article called")
    flag=0

    if topic_name=="caa-nrc":
        # something.fit(tfidf of article)
        article_vec = vectorizer_caa.transform([topic_name])
        cluster_number = km_caa.predict(article_vec)[0]
        # print(cluster_number)
        flag=1

    if topic_name=="coronavirus":
        # something.fit(tfidf of article)
        article_vec = vectorizer_covid.transform([topic_name])
        cluster_number = km_covid.predict(article_vec)[0]
        # print(cluster_number)

    return cluster_number,flag