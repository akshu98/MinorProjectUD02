from __future__ import print_function
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer 
import enchant
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df =pd.read_csv("../data/parsed_news_articles.csv")
df['Text'] = df['Text'].astype(str)


# In[3]:


def preprocessor(text):
    stop_words = stopwords.words('english')
    d = enchant.Dict("en_GB")
    spcl= '[-,;@_!#$%^&*()<>?/\|}{~:''.+""]'
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        token = re.sub(spcl,'',token)
        token = re.sub('[0-9]','',token)
        if re.search('[a-zA-Z]', token):
            token = token.lower()
            if (token!='') and (d.check(token)):
                filtered_tokens.append(token)
    v = [word for word in filtered_tokens if word not in stop_words]
    stems = [lemmatizer.lemmatize(t) for t in v]
    temp = ' '.join(i for i in stems)
    return temp



cleanedArticles=[]

for i in df.index:
    c=preprocessor(df['Text'][i])
    cleanedArticles.append(c)




