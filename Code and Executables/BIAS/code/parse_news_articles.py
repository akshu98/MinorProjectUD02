import pandas as pd
from newspaper import Article

data = pd.read_csv("../data/test_articles.csv")
df= pd.DataFrame(data)

links= df["Links"]
titles=[]
text=[]
cnt=1
for url in links:
	try:
		print(cnt)
		news_article = Article(url)
		news_article.download() 
		news_article.parse()
		news_article.nlp()
		titles.append(news_article.title)
		text.append(news_article.text)
		cnt=cnt+1
	except:
		print(url)
		df.drop(df.loc[df['Links']==url].index, inplace=True)
		print("dropped")
		cnt=cnt+1
    
df["Title"]=titles
df["Text"]=text
df.to_csv("../data/parsed_news_articles.csv",index=None)