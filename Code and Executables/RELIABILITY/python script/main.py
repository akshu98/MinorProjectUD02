import pandas as pd
from newspaper import Article
import preprocess
import Clustering
import Neutrality 
from SourceScore import df
from StrongWordsAnalysis import strong_words_score
from RELIABILITY_PREDICTOR import reliability_finder
import anvil.server

def parse_url(url):
	print("Parser called")
	news_article = Article(url)
	news_article.download() 
	news_article.parse()
	news_article.nlp()
	return news_article.text 



print("HERE")
anvil.server.connect('CV7CBVPVAYRUB2FJ7JCN4BFY-7U44Y5T463ZG2ZRJ') 

#anvil server code
@anvil.server.callable
def reliability_label_predictor(attributes):
	source_name = attributes[0]
	topic_name = attributes[1]
	url = attributes[2]
	print(source_name,topic_name,url)
	article = parse_url(url)
	cleanedArticle=preprocess.preprocessor(article)
	cluster_number,flag = Clustering.cluster_new_article(cleanedArticle,topic_name)
	neutrality_score = Neutrality.neutrality_score_finder(cluster_number,flag,cleanedArticle)
	print(neutrality_score)
	source_score = df.loc[df['Source']==source_name,'SourceScore'].iloc[0]
	print(source_score)
	score_label = strong_words_score(cleanedArticle)
	print(score_label)
	reliability_label = reliability_finder(source_score,neutrality_score,score_label)
	print(reliability_label)
	return reliability_label

anvil.server.wait_forever()
# reliability_label_predictor('The Wire','coronavirus','https://thewire.in/health/india-covid-19-numbers-record-daily-spike')
