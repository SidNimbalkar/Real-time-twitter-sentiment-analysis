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
    def scrapping(self):
        import boto3
        from metaflow import S3
        import pandas as pd

        """ SCRAPE TWEETS HERE """





        self.next(self.preprocessing)

    @conda(libraries={'pandas' : '1.0.1','nltk': '3.4.5','smart_open':'1.9.0'})
    @step
    def preprocessing(self):
        import boto3
        from metaflow import S3
        import pandas as pd
        from nltk import tokenize
        import string
        import nltk
        from nltk.corpus import stopwords
        from smart_open import smart_open

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('words')


        """ PROCESS TWEETS HERE """


        self.next(self.labelling)


    @conda(libraries={'pandas' : '1.0.1','nltk': '3.4.5'})
    @step
    def labelling(self):
        from google.cloud import language_v1
        from google.cloud.language_v1 import enums
        import pandas as pd
        import urllib.request
        import boto3
        from metaflow import S3



        """ Label the dataset using Amazon Comprehend """
        """ Save to S3 bucket """


        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass

if __name__ == '__main__':
    TrainPipeline()
