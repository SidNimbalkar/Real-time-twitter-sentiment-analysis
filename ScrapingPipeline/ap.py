import pandas as pd
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd
import tweepy as tw
import nltk
from nltk.corpus import stopwords
from collections import Counter
import datetime
nltk.download('stopwords')

def scraping():

    consumer_key = "jLdFdm7HLMWSnUDVbRZtejI2b"
    consumer_secret = "9JY4RHN7OB0wVVjxdpvmk4YaGceTwm8T8t0GscatSMIRfDfm7J"
    access_token = "1247295880652296195-rxjW6TaJdsDDH4PMdQvWqQXKZo2W2H"
    access_token_secret = "KJ5j6JBhkWhKUVpoOknO9LDcwNwF0qSq34Dh8FN9fS2q1"
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Gather Tweets
    tweets = []
    public_tweets = api.home_timeline(count=5)
    for tweet in public_tweets:
        tweets.append(tweet.text)

    list = pd.DataFrame({'tweets': tweets})

    common_words = Counter(" ".join(list["tweets"]).split()).most_common(100)
    first_tuple_elements = []

    for a_tuple in common_words:
        # Sequentially access each tuple in `tuple_list`
        first_tuple_elements.append(a_tuple[0])
    search_words = [word for word in first_tuple_elements if word not in stopwords.words('english')]

    for word in search_words:
        tweets = tw.Cursor(api.search,
                           q=word,
                           lang="en",
                           since='2018-04-23').items(5)
    scraped_tweets = [tweet.text for tweet in tweets]

    all_tweets = pd.DataFrame({'tweets': scraped_tweets})
	
    return(all_tweets)
	
def scraping2():
    nltk.download('stopwords')

    consumer_key = "jLdFdm7HLMWSnUDVbRZtejI2b"
    consumer_secret = "9JY4RHN7OB0wVVjxdpvmk4YaGceTwm8T8t0GscatSMIRfDfm7J"
    access_token = "1247295880652296195-rxjW6TaJdsDDH4PMdQvWqQXKZo2W2H"
    access_token_secret = "KJ5j6JBhkWhKUVpoOknO9LDcwNwF0qSq34Dh8FN9fS2q1"
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Gather Tweets
    startDate = datetime.datetime(2020, 4, 9, 0, 0, 0)
    endDate = datetime.datetime(2020, 4, 10, 0, 0, 0)

    user_list = ['cnnbrk', 'nytimes']

    tweets_result = []
    for username in user_list:
        tweets = []
        # print(username)
        tmpTweets = api.user_timeline(username)
        for tweet in tmpTweets:
            if tweet.created_at < endDate and tweet.created_at > startDate:
                tweets.append(tweet)

        while (tmpTweets[-1].created_at > startDate and len(tweets) <= 3200):
            # print("Last Tweet @", tmpTweets[-1].created_at, " - fetching some more")
            tmpTweets = api.user_timeline(username, max_id=tmpTweets[-1].id)
            for tweet in tmpTweets:
                if tweet.created_at < endDate and tweet.created_at > startDate:
                    tweets.append(tweet)

        for i in tweets:
            tweets_result.append(i)

    scraped_tweets = [tweet.text for tweet in tweets_result]

    df_users = pd.DataFrame({'Tweet': scraped_tweets})
	
    return(df_users)

def preprocess_tweet(text):
    # Check characters to see if they are in punctuation
    nopunc = text
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
    nopunc = re.sub(r'http\S+', '', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '', nopunc)
    # remove the # in #hashtag
    nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)

    return (nopunc.strip())
		
