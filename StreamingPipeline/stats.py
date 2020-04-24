from kafka import KafkaConsumer
import json
from textblob import TextBlob
from kafka import KafkaProducer


consumer1 = KafkaConsumer('test')
favorite_count_min=100000000
favorite_count_max=0
favorite_count_sum=0
favorite_count_n=0
favorite_count_range=(100000000,0)
favorite_count_average=0
followers_count_min=100000000
followers_count_max=0
followers_count_sum=0
followers_count_n=0
followers_count_range=[100000000,0]
followers_count_average=0

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
tweet_dict= {}

for msg in consumer1:
    tweet_text_json=json.loads(msg.value.decode('utf-8'))
    try:
        tweet_favorite_count = tweet_text_json['retweeted_status']['favorite_count']
        tweet_followers_count = tweet_text_json['retweeted_status']['user']['followers_count']



        tweet_dict['topic']='key'
        tweet_dict['timestamp_ms']=tweet_text_json['timestamp_ms']
        tweet_dict['tweet_favorite_count']=tweet_favorite_count
        tweet_dict['followers_count']= tweet_followers_count
        tweet_dict['text'] = tweet_text_json['text']

        #producer.send('key'+'_'+'stats', bytes(json.dumps(tweet_dict),'utf-8'))
        producer.send('stat', bytes(json.dumps(tweet_dict),'utf-8'))
        print(json.dumps(tweet_dict))
    except KeyError:
        print("No retweet status")
