from kafka import KafkaConsumer
consumer = KafkaConsumer('sample1')
for message in consumer:
    print (message)
