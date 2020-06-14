#!/usr/bin/env python
# coding: utf-8




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

#anvil 
import anvil.server

lemmatizer = WordNetLemmatizer()





df =pd.read_csv("../data/parsed_news_articles.csv")
df['Text'] = df['Text'].astype(str)





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





len(cleanedArticles)





df['Cleaned_text']=cleanedArticles





test_articles = pd.read_csv("../data/parsed_test_articles.csv")
cleanedTestArticles=[]

for i in test_articles.index:
    c=preprocessor(test_articles['Text'][i])
    cleanedTestArticles.append(c)
    
test_articles['Cleaned_text']=cleanedTestArticles





def vectorize_sim_search(query, datai):
    cos_sim=[]
    tfidf_vectorizer = TfidfVectorizer()
    dataMatrixAll = []
    similarity_dict=dict()
    for i in range(len(query)):
        query_i = query[i]
        data = [query[i]]+cleanedArticles
        lk=test_articles['Links'][i]
        similarity_dict[lk]=list()
        data = [query[i]]+datai
        dataMatrix = {}
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        vals = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0]
        for j in range(len(data)-1):
            dataMatrix[df.loc[df['Cleaned_text'] == datai[j], 'Links'].iloc[0]]=vals[j+1]
        
        key_list = list(dataMatrix.keys())
        val_list = list(dataMatrix.values())
        sorted_vals = sorted(vals,reverse=True)[1:6]
        for i in sorted_vals:
            temp=key_list[val_list.index(i)]
            similarity_dict[lk].append(temp)
    return similarity_dict





def sentiment_scores(text):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text)
    return sentiment_dict['compound']   





df["sentiment"]=""
for i in range(len(cleanedArticles)):
    article_title = df['Title'].iloc[i]
    article_body = cleanedArticles[i]
    df['sentiment'][i] = sentiment_scores(article_body)

d=dict()
keys = df['Source'].unique()
for key in keys:
    d[key]=list()
for i in range(len(df)):
    d[df['Source'][i]].append(df['Cleaned_text'][i])
comparison_dict=dict()
links = list(test_articles['Links'])
for i in range (len(test_articles)):
    comparison_dict[links[i]]=list()
for source,articles in d.items():
    # print(source)
    d2 = vectorize_sim_search(cleanedTestArticles,articles)
    ke=list(d2.keys())
    for key in ke:
        comparison_dict[key].append(d2[key][0])

k = list(comparison_dict.keys())
sourcesl = list(d.keys())
avg_sc = [0]*27
for link in k:
    for i in range(len(comparison_dict[link])):
        avg_sc[i]=avg_sc[i]+df.loc[df['Links'] == comparison_dict[link][i], 'sentiment'].iloc[0]
for i in range(len(avg_sc)):
    avg_sc[i]=avg_sc[i]/29
bias_list=[0]*27
for i in range(len(avg_sc)):
    if(avg_sc[i]<-0.25):
        bias_list[i]='Left'
    elif(avg_sc[i]>0.05):
        bias_list[i]='Right'
    else:
        bias_list[i]='Center'
bias_df = pd.DataFrame(list(zip(sourcesl, bias_list)),columns =['Source', 'Bias'])
# print(bias_df)






# connecting to anvil server
anvil.server.connect('DD635TLLOAUOC42HNR5XRTIM-7U44Y5T463ZG2ZRJ')

#anvil server code
@anvil.server.callable
def bias_label_predictor(source_name):
    print(source_name)
    bias_label = bias_df.loc[bias_df['Source']==source_name,'Bias'].iloc[0]
    print(bias_label)

    return bias_label

anvil.server.wait_forever()
