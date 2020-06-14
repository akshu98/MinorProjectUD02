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

# df =pd.read_csv("../data/parsed_news_articles.csv")
# df['Text'] = df['Text'].astype(str)

def preprocessor(text):
    print("Preprocessor called")
    stop_words = stopwords.words('english')
    d = enchant.Dict("en_GB")
    spcl_words = ['also','said','olive','glossary','pe','butterfly','lip']
    spcl_names = ['bjp','shaheen','bagh',
                  'caa','nrc','covid','coronavirus','corona','virus','pm']
#     'modi','pm','gandhi','rahul','amit','shah','sonia','congress',
    spcl= '[-,;@_!#$%^&*()<>?/\|}{~:''.+""]'
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        token = re.sub(spcl,'',token)
        token = re.sub('[0-9]','',token)
        if re.search('[a-zA-Z]', token):
            token = token.lower()
            if token in spcl_names:
                filtered_tokens.append(token)
            if (token!='') and (d.check(token)) and (token not in spcl_words):
                filtered_tokens.append(token)
    v = [word for word in filtered_tokens if word not in stop_words]
    stems = [lemmatizer.lemmatize(t) for t in v]
    temp = ' '.join(i for i in stems)
    return temp


# In[4]:


# cleanedArticles=[]

# for i in df.index:
#     c=preprocessor(df['Text'][i])
#     cleanedArticles.append(c)


# df['Cleaned_text']=cleanedArticles
# df.to_csv("../data/parsed_cleaned.csv",index=None)
