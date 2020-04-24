## Scraping Pipeline

This Pipeline scrapes historic tweets using tweepy, Pre-processes the said tweets and uses Amazon Comprehend API to generate a sentiment score for every tweet. It then stores the labelled dataset on a S3 bucket.

## Pre requisite

- The requirements.txt in the main Readme should be installed

- To access the Amazon Comprehend, you'll need an AWS account, with the account keys configured 

- Since the pipeline is designed using Metaflow, you'll need to run it on Linux, Mac OS X, or other Unix-like OS (Windows is not supported)

## Run Instructions 

```
python annotation_pipeline.py --environment=conda run
```

This pipeline will write it's result to an output bucket on AWS S3 which is specified and configureed in the pipeline.


#### The pipeline model is shown below:

![alt text](https://github.com/SidNimbalkar/CSYE7245FinalProject/blob/master/Images/pipe1-2.png)
