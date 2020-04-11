import tweepy
import datetime
import pandas as pd

# credentials from https://apps.twitter.com/
consumerKey = "w9Dj65V9mr4v1OCml72rlersQ"
consumerSecret = "PRpUhWbRvGQ1o3xkdalGUGlB1cwjNSmjFshba09e0nKdafcuHY"
accessToken = "838352454517424128-aUaIuR6IgUyUZzwdOZ3JvVPlaDoJrYw"
accessTokenSecret = "ENAPFU3NUDYHB79kdGwjPGlDYQFhTUh0BIYElcCX2DTI5"

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth,wait_on_rate_limit=True)


startDate = datetime.datetime(2010,4,10,0,0,0)
endDate = datetime.datetime(2020,4,10,0,0,0)


user_list = ['cnnbrk','nytimes','cnn','bbcbreaking','bbcworld','theeconomist','reuters','wsj','time','abc','telegraph','cbcnews','independent','rt_com','guardiannews',
             'BarackObama','justinbieber','katyperry','rihanna','Cristiano','ladygaga','YouTube','KimKardashian','TheEllenShow','ArianaGrande',
             'skynewsbreak','huffpost','ap','jimmyfallon','shakira','narendramodi','britneyspears','Twitter','selenagomez','jtimberlake','BillGates',
             'neymarjr','JLo','Oprah','iamsrk','SrBachchan']

tweets_result = []
for username in user_list:
  tweets = []
  #print(username)
  tmpTweets = api.user_timeline(username)
  for tweet in tmpTweets:
    if tweet.created_at < endDate and tweet.created_at > startDate:
      tweets.append(tweet)

  while(tmpTweets[-1].created_at > startDate and len(tweets)<=3200):
    #print("Last Tweet @", tmpTweets[-1].created_at, " - fetching some more")
    tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
    for tweet in tmpTweets:
      if tweet.created_at < endDate and tweet.created_at > startDate:
        tweets.append(tweet)
		
  for i in tweets:
    tweets_result.append(i)
     
scraped_tweets = [tweet.text for tweet in tweets_result]

df_users = pd.DataFrame({'Tweet' : scraped_tweets})
