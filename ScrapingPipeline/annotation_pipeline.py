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
    def scraping(self):
        import pandas as pd
        import tweepy as tw
        import nltk
        from nltk.corpus import stopwords
        from collections import Counter

        nltk.download('stopwords')

        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
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
    def scraping2(self):
        import pandas as pd
        import tweepy as tw
        import nltk
        from nltk.corpus import stopwords
        from collections import Counter
        import datetime

        nltk.download('stopwords')

        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
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


        self.next(self.preprocessing)


    @conda(libraries={'pandas' : '1.0.1','nltk': '3.4.5'})
    @step
    def preprocessing(self):
        import pandas as pd
        from nltk.tokenize import word_tokenize
        import string
        import re
        List_clean = []
        
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
            return (nopunc)

        for x in self.combined_df['Tweet']:
            List_clean.append(preprocess_tweet(x))

        self.clean_df = pd.DataFrame(List_clean,columns=['tweet'])
        
        self.next(self.labelling)


    @conda(libraries={'pandas' : '1.0.1'})
    @step
    def labelling(self):
        import pandas as pd
        import boto3
        from metaflow import S3
        def create_sentiment_aws(row):
            """Uses AWS Comprehend to Create Sentiments on a DataFrame"""

            try:
              comprehend = boto3.client(service_name='comprehend', region_name="us-east-2")
              payload = comprehend.detect_sentiment(Text=row, LanguageCode='en')
              sentiment = payload['Sentiment']
            except Exception:
              print("Size exceeded:  Fail")
              return None
            return sentiment

        def apply_sentiment_aws(df, column="text"):
            """Uses Pandas Apply to Create Sentiment Analysis"""
            df['Sentiment'] = df[column].apply(create_sentiment_aws)
            return df

        L_aws = []

        for x in self.clean_df['tweet']:
            comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
            comp_str = comprehend.detect_sentiment(Text=x, LanguageCode='en')
            if comp_str['Sentiment'] == 'POSITIVE':
                L_aws.append([x,2])
            elif comp_str['Sentiment'] == 'NEGATIVE':
                L_aws.append([x,0])
            elif comp_str['Sentiment'] == 'NEUTRAL':
                L_aws.append([x,1])

        f_sentiment = pd.DataFrame(L_aws, columns = ['Tweet','Score'])
        
        final_sentiment = pd.DataFrame({'label':f_sentiment[1],'tweet': f_sentiment[0].replace(r'\n', ' ', regex=True)})

        final_sentiment.to_csv('labelledtweets.tsv', sep='\t', index=False, header=False)

        with S3(s3root='s3://sentstorage/') as s3:
            s3.put_files([('labelledtweets.tsv','labelledtweets.tsv')])



        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass

if __name__ == '__main__':
    TrainPipeline()
