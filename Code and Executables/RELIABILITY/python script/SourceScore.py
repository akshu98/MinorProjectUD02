#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv("../data/SourceInfo.csv")


# In[3]:


df.head()


# In[4]:


df["SourceScore"]=""

d ={"High":3,"Low":1,"Medium":2}


# In[5]:


for i in df.index:
    df["SourceScore"][i]=0.2*df['Wikipedia'][i]+0.01*(2020-df['Year'][i])+0.1*df['Twitter'][i]+0.05*df['Verified'][i]+0.37*d[df['Popularity'][i]]+0.2*df['History'][i]+0.07*df['Controversy'][i]


# In[6]:


df.sort_values(by="SourceScore")


# In[7]:


a=max(df['SourceScore'])
for i in range(len(df)):
    b=df['SourceScore'][i]/a
    df['SourceScore'][i]=b


# In[8]:


df.sort_values(by="SourceScore")


# In[9]:


df.to_csv("../data/SourceInfo.csv",index=None)

