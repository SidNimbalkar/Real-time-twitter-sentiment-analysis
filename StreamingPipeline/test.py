from kafka import KafkaConsumer
import json

consumer = KafkaConsumer('sample1')
for msg in consumer:
    tweet_text_json=json.loads(msg.value.decode('utf-8'))
    print(tweet_text_json)
