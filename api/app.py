# Using flask to make an api
from flask import Flask, jsonify, request
import json
import pandas as pd
import os
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf


# creating a Flask app
app = Flask(__name__)

# on the terminal type: curl http://127.0.0.1:5000/
@app.route('/', methods = ['GET','POST'])
def home():
    if(request.method == 'POST' or request.method == 'GET'):

        data = "hello world"
        return jsonify({'data': data})


@app.route('/predict', methods = ['POST', 'GET']) #/result route, allowed request methods; POST, and GET
def predict():
    if request.method == 'POST':
        text = str(request.get_data())#as_text=True

        model = BertForSequenceClassification.from_pretrained("model")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_idstest = []
        attention_maskstest = []

        encoded_dicttest = tokenizer.encode_plus(text, add_special_tokens=True, max_length=200, pad_to_max_length=True,
                                                 return_attention_mask=True, return_tensors='pt', )

        input_idstest.append(encoded_dicttest['input_ids'])

        attention_maskstest.append(encoded_dicttest['attention_mask'])

        # Convert the lists into tensors.
        input_idstest = torch.cat(input_idstest, dim=0)
        attention_maskstest = torch.cat(attention_maskstest, dim=0)

        outputs = model(input_idstest, token_type_ids=None, attention_mask=attention_maskstest)
        out = outputs[0]
        sc = torch.max(out[0])
        score = sc.data.numpy()
        b = torch.argmax(out[0])
        idx = np.unravel_index(b, out[0].shape)
        sentiment = idx[0]
        if sentiment == 2:
            sent = str(score)
        elif sentiment == 0:
            sent = str(-1 * score)
        elif sentiment == 1:
            sent = '0'

        return (sent)


# driver function
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
