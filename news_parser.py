import pandas as pd
from newspaper import Article

f = open("news articles/articles.txt","r")
LOA = [line.rstrip() for line in f]
df = pd.DataFrame(columns = ['Title','Text','URL','Cleaned_text'])
for url in LOA: 
    try:
        news_article = Article(url, language="en")     
        news_article.download() 
        news_article.parse() 
        news_article.nlp() 
        df2={'Title':news_article.title,'Text':news_article.text,'URL':url}
        df=df.append(df2,ignore_index=True)
    
    except Exception:
        print(url)
    print(url,"PARSED")
        
        
df.to_csv("news articles/parsed_news_articles.csv",index=None)