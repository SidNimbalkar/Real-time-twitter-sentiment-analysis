import pandas as pd  
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re

frames = [df_users, df_kaggle, df_random]
df_tweets = pd.concat(frames,ignore_index= True)

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
	
clean_tweet_texts = []
for i in range(len(df_tweets.index)):
  clean_tweet_texts.append(tweet_cleaner(df_tweets['Tweet'][i]))

clean_tweets = pd.DataFrame(clean_tweet_texts,columns=['Tweet'])
clean_tweets = clean_tweets.dropna()
# write clean_tweets to S3 bucket


