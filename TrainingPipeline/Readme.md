
## Training Pipeline 

In this pipeline we train a ML model using transfer learning with BERT(Bidirectional Encoder Representations from Transformers), we train this model on the labeled dataset that we generated in the Annotation Pipeline using historic tweets. 

We used the pickled model to create a Flask API that takes in a json input and generates a sentiment and sentiment score.

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


## Testing 

The bertfine python notebook was used to train and test the model using Google Colab. The GPU was leveraged to improve training speed.

