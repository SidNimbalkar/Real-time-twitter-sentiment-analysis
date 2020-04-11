"""
    In this pipeline we will scrape tweets and create a labelled dataset using Amazon comprehend API
    We then write this dataset to the S3 bucket and use it to train a machine learning model in our next pipeline
"""

from metaflow import FlowSpec, step, retry, catch, batch, IncludeFile, Parameter, conda, conda_base,S3
import boto3
import csv

def get_python_version():

    import platform
    versions = {'3' : '3.7.4'}
    return versions[platform.python_version_tuple()[0]]


@conda_base(python=get_python_version())
class TrainPipeline(FlowSpec):

    @conda(libraries={'pandas' : '1.0.1'})
    @step
    def start(self):
        print ("Scraping has commenced!")
        self.next(self.scrapping,self.scrapping2)


    @conda(libraries={'pandas' : '1.0.1','tweepy':'3.8.0','nltk': '3.4.5'})
    @step
    def scrapping(self):
        import pandas as pd
        import tweepy as tw
        import nltk
        from nltk.corpus import stopwords
        from collections import Counter

        nltk.download('stopwords')

        consumer_key = "w9Dj65V9mr4v1OCml72rlersQ"
        consumer_secret = "PRpUhWbRvGQ1o3xkdalGUGlB1cwjNSmjFshba09e0nKdafcuHY"
        access_token = "838352454517424128-aUaIuR6IgUyUZzwdOZ3JvVPlaDoJrYw"
        access_token_secret = "ENAPFU3NUDYHB79kdGwjPGlDYQFhTUh0BIYElcCX2DTI5"
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth,wait_on_rate_limit=True)

        #Gather Tweets
        tweets = []
        public_tweets = api.home_timeline(count = 5)
        for tweet in public_tweets:
          tweets.append(tweet.text)

        list = pd.DataFrame({'tweets' : tweets})


        common_words = Counter(" ".join(list["tweets"]).split()).most_common(100)
        first_tuple_elements = []

        for a_tuple in common_words:
        #Sequentially access each tuple in `tuple_list`
            first_tuple_elements.append(a_tuple[0])
        search_words = [word for word in first_tuple_elements if word not in stopwords.words('english')]

        for word in search_words:
          tweets = tw.Cursor(api.search,
                      q=word,
                      lang="en",
                      since='2018-04-23').items(5)
        scraped_tweets = [tweet.text for tweet in tweets]

        self.all_tweets = pd.DataFrame({'tweets' : scraped_tweets})


        self.next(self.join)


    @conda(libraries={'pandas' : '1.0.1','tweepy':'3.8.0','nltk': '3.4.5'})
    @step
    def scrapping2(self):
        import pandas as pd
        import tweepy as tw
        import nltk
        from nltk.corpus import stopwords
        from collections import Counter
        import datetime

        nltk.download('stopwords')

        consumer_key = "w9Dj65V9mr4v1OCml72rlersQ"
        consumer_secret = "PRpUhWbRvGQ1o3xkdalGUGlB1cwjNSmjFshba09e0nKdafcuHY"
        access_token = "838352454517424128-aUaIuR6IgUyUZzwdOZ3JvVPlaDoJrYw"
        access_token_secret = "ENAPFU3NUDYHB79kdGwjPGlDYQFhTUh0BIYElcCX2DTI5"
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth,wait_on_rate_limit=True)

        #Gather Tweets
        startDate = datetime.datetime(2020,4,9,0,0,0)
        endDate = datetime.datetime(2020,4,10,0,0,0)


        user_list = ['cnnbrk','nytimes']

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

        self.df_users = pd.DataFrame({'Tweet' : scraped_tweets})


        self.next(self.join)

    @conda(libraries={'pandas' : '1.0.1'})
    @step
    def join(self,inputs):
        import pandas as pd

        self.combined_df = pd.DataFrame()
        self.combined_df = pd.concat([inputs.scrapping2.df_users, inputs.scrapping.all_tweets], ignore_index=True)
        print (inputs.scrapping.all_tweets)





        self.next(self.preprocessing)


    """@conda(libraries={'pandas' : '1.0.1','nltk': '3.4.5','smart_open':'1.9.0'})
    @step
    def preprocessing(self):
        import re
        import pandas as pd
        from nltk import tokenize
        import string
        import nltk
        from nltk.corpus import stopwords

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('words')
        nltk.download('wordnet')


        def remove_punct(text):
            text  = "".join([char for char in text if char not in string.punctuation])
            text = re.sub('[0-9]+', '', text)
            return text
        self.all_tweets['Tweet_punct'] = self.all_tweets['tweets'].apply(lambda x: remove_punct(x))

        def tokenization(text):
          text = re.split('\W+', text)
          return text
        self.all_tweets['Tweet_tokenized'] = self.all_tweets['Tweet_punct'].apply(lambda x: tokenization(x.lower()))

        stopword = nltk.corpus.stopwords.words('english')
        def remove_stopwords(text):
            text = [word for word in text if word not in stopword]
            return text
        self.all_tweets['Tweet_nonstop'] = self.all_tweets['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))

        ps = nltk.PorterStemmer()

        def stemming(text):
            text = [ps.stem(word) for word in text]
            return text
        self.all_tweets['Tweet_stemmed'] = self.all_tweets['Tweet_nonstop'].apply(lambda x: stemming(x))

        wn = nltk.WordNetLemmatizer()

        def lemmatizer(text):
            str = " "
            text = [wn.lemmatize(word) for word in text]
            return str.join(text)
        self.all_tweets['Tweet_lemmatized'] = self.all_tweets['Tweet_nonstop'].apply(lambda x: lemmatizer(x))


        self.next(self.labelling)


    @conda(libraries={'pandas' : '1.0.1','nltk': '3.4.5'})
    @step
    def labelling(self):
        import json
        import pandas as pd
        import boto3
        key = 'AKIAJG5BRIEPKOSCS5WQ'
        sec_key = 'LmFTVSfFhQgJTuL/hHlKHw2XGpYXI5ShjvqCi7t6'
        json_file = []

        comprehend = boto3.client(service_name = 'comprehend', aws_access_key_id=key, aws_secret_access_key=sec_key, region_name='us-east-1')

        clean_tweet = self.all_tweets['Tweet_lemmatized'].head(10)

        print('Calling DetectSentiment')
        for tweet in clean_tweet:
          print(json.dumps(comprehend.detect_sentiment(Text=tweet, LanguageCode='en'), sort_keys=True, indent=4))
        print('End of DetectSentiment\n')

        scores = []
        sentiment_score = []
        for x in range(0,25000):
          scores.append(json_file[0:25000][x]['SentimentScore'])

        for i in range(0,25000):
          all_values = scores[i].values()
          sentiment_score.append(max(all_values))

        label = []
        for n in range(0,25000):
          label.append(json_file[0:25000][n]['Sentiment'])

        label_tweets = pd.DataFrame({'Tweets' : clean_tweet, 'label' : label, 'Score' : sentiment_score})

        print(label_tweets)
        self.next(self.end)"""

    @step
    def end(self):
        """
        End the flow.
        """
        pass

if __name__ == '__main__':
    TrainPipeline()
