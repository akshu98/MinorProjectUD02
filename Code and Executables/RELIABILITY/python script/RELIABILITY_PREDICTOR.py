#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import Counter


# In[2]:


# source_info = pd.read_csv('../data/SourceInfo.csv')
# caa_c0=pd.read_csv("../data/caa_nrc_cluster_0.csv")
# caa_c1=pd.read_csv("../data/caa_nrc_cluster_1.csv")
# caa_c2=pd.read_csv("../data/caa_nrc_cluster_2.csv")
# covid_c0=pd.read_csv("../data/covid_cluster_0.csv")
# covid_c1=pd.read_csv("../data/covid_cluster_1.csv")
# covid_c2=pd.read_csv("../data/covid_cluster_2.csv")
# original_parsed_data = pd.read_csv("../data/parsed_cleaned.csv")
# strong_words_data = pd.read_csv("../data/StrongWordsAnalysis.csv")


# # In[24]:


# caa_c0.head()


# # In[3]:


# Counter(original_parsed_data['Source'])


# # In[4]:


# articles_copy = caa_c0
# articles_copy = articles_copy.append([caa_c1,caa_c2,covid_c0,covid_c1,covid_c2])


# # In[5]:


# articles_copy


# # In[6]:


# articles = []
# for i in original_parsed_data.index:
#     source=original_parsed_data['Source'].iloc[i]
#     title=original_parsed_data['Title'].iloc[i]
#     topic=original_parsed_data['Topic Name'].iloc[i]
#     text=original_parsed_data['Cleaned_text'].iloc[i]
#     source_reliability=source_info.loc[source_info['Source']==source,'SourceScore'].iloc[0]
#     strong_words_score = strong_words_data.loc[strong_words_data['Title']==title,'strongP'].iloc[0]
#     neutrality_score=articles_copy['neutrality'].iloc[i]
#     articles.append([source,title,topic,text,source_reliability,strong_words_score,neutrality_score])


# # In[7]:


# articles_data = pd.DataFrame(articles,columns=["Source","Title","Topic Name","Cleaned Text","Source Score","Strong Words Score","Neutrality Score"])
# articles_data.head()


# # In[8]:


# articles_data['Reliability Score']=""
# for i in articles_data.index:
#     score = (0.35)*articles_data['Source Score'][i]+(0.35)*articles_data['Neutrality Score'][i]+(0.3)*articles_data['Strong Words Score'][i]
#     articles_data['Reliability Score'][i] = round(score,3)


# # In[9]:


# a=max(articles_data['Reliability Score'])
# for i in range(len(articles_data)):
#     b=articles_data['Reliability Score'][i]/a
#     articles_data['Reliability Score'][i]=b


# # In[10]:


# articles_data['Reliability Label']=""
# for i in articles_data.index:

#     if articles_data['Reliability Score'][i]>=0.6:
#         articles_data['Reliability Label'][i]='Reliable'
#     else:
#         articles_data['Reliability Label'][i]='Not Reliable'


# # In[11]:


# articles_data.head()


# # In[12]:


# l1 = list(articles_data['Source Score'])
# l2 = list(articles_data['Strong Words Score'])
# l3 = list(articles_data['Neutrality Score'])
# l4 = list(articles_data['Reliability Score'])



# # In[13]:


# reliable_articles=articles_data[articles_data['Reliability Label']=='Reliable']
# unreliable_articles=articles_data[articles_data['Reliability Label']=='Not Reliable']
# s1=Counter(reliable_articles['Source'])
# s2=Counter(unreliable_articles['Source'])




# # In[14]:


# Counter(articles_data['Reliability Label'])
# s1=dict(s1)
# s2=dict(s2)
# s2


# # In[15]:


# Counter(articles_data['Reliability Label'])


# # In[16]:


# source_info['Reliability Score']=""
# for i in range(len(source_info)):
#     if source_info['Source'][i] not in s1:
#         source_info['Reliability Score'][i] = 0
#     elif source_info['Source'][i] not in s2:
#         source_info['Reliability Score'][i] = 1
#     else:
#         source_info['Reliability Score'][i] = s1[source_info['Source'][i]]/(s2[source_info['Source'][i]]+s1[source_info['Source'][i]])


# # In[17]:


# source_info.sort_values(by='Reliability Score')





# source_info['Reliability Label']=""
# for i in range(len(source_info)):

#     if source_info['Reliability Score'][i]>0.6:
#         source_info['Reliability Label'][i]='Reliable'
#     else:
#         source_info['Reliability Label'][i]='Not Reliable'



# # In[20]:


# # print("Number of reliable sources")
# # Counter(source_info['Reliability Label'])


# # In[23]:


# articles_data.head()



def reliability_finder(source_score,neutrality_score,strong_words_score):
    # score = (0.35)*source_score+(0.35)*neutrality_score+(0.3)*strong_words_score
    score = (0.35)*source_score
    score = score + (0.35)*neutrality_score
    score = score + (0.3)*strong_words_score
    if score>0.6:
        reliability_label = "Reliable"
    else:
        reliability_label = "Not Reliable"

    return reliability_label