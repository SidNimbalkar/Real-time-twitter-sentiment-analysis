from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import KafkaProducer
from kafka.client import SimpleClient
from kafka.producer import SimpleProducer
from kafka import KafkaConsumer
import json
import os
import sys
# client = SimpleClient("127.0.0.1:9092")
# producer = SimpleProducer(client)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
#consumer_key= 'tYCzKOoBOZft1oy0cC85l0mxU'
#consumer_secret= 'tS2e6cMQv60ZtvoBaAXenLrkr26oSg8c22fyrcjSUvHZGNyIuS'
#access_token= '4808614249-s758NEUKUDHjmspTSqqak1zSNkmgOI3mOyio0tv'
#access_token_secret= 'JBW4qGXXs3MAXCgwplERYOqnCiJylN3uF8oKlPWn3r7wn'

consumer_key = "GDh3SVKPFjWIUOmdUqOvHnjCq"
consumer_secret = "pi1yUfsx8h9DdTVoovJtMq3AM7CUOynqb5uKBzveSf5g9bg3TD"
access_token = "111887870-BMIWdaFCLcXGaHtSjEMckkyc49osDopzdnlRgN6m"
access_token_secret = "5hAaBQDdgo3Jryp6IDAbcRQX3OUWbkRL6poBDEfxwJYK5"


class StdOutListener(StreamListener):
    key=''
    def set_keys(self,key):
        self.key = key
    def on_data(self, data):
        if self.key in str(data):

            producer.send('tweetz', bytes(data,'utf-8'))
        print(data)
        return True
    def on_error(self, status):
        print(status)
if __name__ == '__main__':
    key='trump'

    l = StdOutListener()
    l.set_keys(key)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=[key],languages=["en"])
