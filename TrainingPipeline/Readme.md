
## Training Pipeline 

In this pipeline we train a ML model BERT Model, we train this model on the labeled dataset that we generated in the Annotation Pipeline using historic tweets. 

We will use the pickled modle to create a flask api that will take in a json input and create a sentiment.

## Pre requisite

- The requirements.txt in the main Readme should be installed
- Run the scraping pipeline (annotation_pipeline) beforehand, so you have a labelled dataset, which is the input for our training pipeline.
- Since the pipeline is designed using Metaflow, you'll need to run it on Linux, Mac OS X, or other Unix-like OS (Windows is not supported)

## Run Instructions 

```
python training.py run
```

This is the model for our training pipeline:
![alt text]()


### Testing 

The bertfine python notebook was used to train and test the model using Google Colab. The GPU was leveraged to improve training speed.

