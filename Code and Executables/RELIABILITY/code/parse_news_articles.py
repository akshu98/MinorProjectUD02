import pandas as pd
from newspaper import Article

data = pd.read_csv("../data/articles_links.csv")
df= pd.DataFrame(data)

links= df["Link"]
titles=[]
text=[]
cnt=1
urls = []
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
		print("problem")
		urls.append([df.loc[df['Link']==url,'Source'].iloc[0],url])
		df.drop(df.loc[df['Link']==url].index, inplace=True)
		print("added to reparse df")
		cnt=cnt+1
    
df["Title"]=titles
df["Text"]=text
df.to_csv("../data/parsed_news_articles.csv",index=None)

reparse_df = pd.DataFrame(urls,columns=['Source','Link'])
reparse_df.to_csv('../data/reparse_data.csv')