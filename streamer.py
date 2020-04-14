import re
import tweepy
import mysql.connector
import pandas as pd
from textblob import TextBlob

API_KEY = ""
API_SECRET_KEY = ""
ACCESS_TOEKN = ""
ACCESS_TOKEN_SECRET = ""

val = input("Enter topic: ")

TRACK_WORDS = [val]
TABLE_NAME = val
TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at DATETIME, text VARCHAR(255), \
            polarity INT, subjectivity INT, user_created_at VARCHAR(255), user_location VARCHAR(255), \
            user_description VARCHAR(255), user_followers_count INT, longitude DOUBLE, latitude DOUBLE, \
            retweet_count INT, favorite_count INT"

class MyStreamListener(tweepy.StreamListener):
    #Tweets are known as “status updates”. So the Status class in tweepy has properties describing the tweet.
    #https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html


    def on_status(self, status):
        #Extract info from tweets


        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        id_str = status.id_str
        created_at = status.created_at
        text = deEmojify(status.text)    # Pre-processing the text
        sentiment = TextBlob(text).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity

        user_created_at = status.user.created_at
        user_location = deEmojify(status.user.location)
        user_description = deEmojify(status.user.description)
        user_followers_count =status.user.followers_count
        longitude = None
        latitude = None
        if status.coordinates:
            longitude = status.coordinates['coordinates'][0]
            latitude = status.coordinates['coordinates'][1]

        retweet_count = status.retweet_count
        favorite_count = status.favorite_count

        #print(status.text)
        #print("Long: {}, Lati: {}".format(longitude, latitude))

    def on_error(self, status_code):
        #Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.

        if status_code == 420:
            # return False to disconnect the stream
            return False


def clean_tweet(self, tweet):
    #Use sumple regex statemnents to clean tweet text by removing links and special characters

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def deEmojify(text):
    #Strip all non-ASCII characters to remove emoji characters

    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None

#Authenticate TwitterAPI
auth  = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOEKN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#Start Stream
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
myStream.filter(languages=["en"], track = TRACK_WORDS)
