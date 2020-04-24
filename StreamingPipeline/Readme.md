## Streaming Pipeline

This is a Kafka pipeline to stream and process live data. We scrape live tweets into kafka using tweepy, 
we then use some pre-processing to clean the tweets, we use the MicroService we created to get sentiment scores for our tweets, 
the kafka events that are generated pack the timestamp, tweet text, all the tweet related information like retweet count, favorite_count, etc.
We then send these events to Druid where they get flattend and stored into a live table, We then use the tool turnilo on druid dataset to visualize the live stream.


## Pre requisite
- The requirements.txt in the main Readme should be installed
- You must have a zookeeper and kafka server running


## Run Instructions 

1. Start Zookeeper <br/>
`
 bin/zookeeper-server-start.sh config/zookeeper.properties
`

2. Start Kafka <br/>
`
bin/kafka-server-start.sh config/server.properties
`

3. Create two kafka topics<br/>
`
kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tweet
`
<br/>
`
kafka-topics --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic druid
`

4. Run our Kafka Prodcuer<br/>
`
python producer.py
`

5. Run our Kafka Consumer<br/>
`
python consumer_full.py
`

We have dockerized and created a kubernetes cluster for the pipeline for ease of use</br>
Our kafka cluster can be run using the following command: </br>
`
docker-compose up -d
`

Now that we have our kafka stream running, we can load the data into druid for visualization and analysis
 
### Kafka Architecture

![alt text](https://github.com/SidNimbalkar/CSYE7245FinalProject/blob/master/Images/Kafka.png)


